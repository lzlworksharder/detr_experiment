from torch.cuda import is_available
import os
import numpy as np
from transformers.image_transforms import center_to_corners_format
import torch
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# training info
def print_model_info(model):
    '''
    print device and model info before training
    '''
    device = model.device
    gpu_available = is_available()
    tpu_available = "XLA_FLAGS" in os.environ  # 判断环境变量是否包含TPU信息
    local_rank = int(os.getenv('LOCAL_RANK', 0))  # 从环境变量获取LOCAL_RANK
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '[0]')  # 获取当前可见设备

    # 模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    estimated_model_size_mb = total_params * 4 / 1024 / 1024  # 假设每个参数占4字节

    # 打印信息
    print(f"GPU available: {gpu_available} (cuda), model device: {device}")
    print(f"TPU available: {tpu_available}, using: {0 if not tpu_available else 1} TPU cores")
    print(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}\n")
    
    # 打印模型参数概览
    print(f" | Name      | Type                             | Params")
    print("-" * 66)
    print(f" | model     | {model.__class__.__name__.ljust(32)} | {total_params / 1e6:.1f} M")
    print("-" * 66)
    print(f"{trainable_params / 1e6:.1f} M    Trainable params")
    print(f"{non_trainable_params/ 1e6:.1f} M    Non-trainable params")
    print(f"{total_params / 1e6:.1f} M    Total params")
    print(f"{estimated_model_size_mb:.3f}   Total estimated model params size (MB)")

# pre-processing
def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }
def augment_and_transform_batch(examples,transform,image_processor,return_pixel_mask=False):
    images=[]
    annotations=[]
    for image_id,image,objects in zip(examples['image_id'],examples['image'],examples['objects']):
        image = np.array(image.convert('RGB'))
        try:
            output = transform(image=image,bboxes=objects['bbox'],category=objects['category'])
        except ValueError:
            continue
        images.append(output['image'])
        formatted_annotations = format_image_annotations_as_coco(image_id,output['category'],objects['area'],output['bboxes'])
        annotations.append(formatted_annotations)

    result = image_processor(images=images,annotations=annotations,return_tensors='pt')
    if not return_pixel_mask:
        result.pop('pixel_mask',None)
    return result 

# post-processing
def convert_bbox_yolo_to_pascal(boxes,image_size):
    height,width = image_size
    boxes = center_to_corners_format(boxes)
    return boxes*torch.tensor([[width,height,width,height]],device=boxes.device)

@dataclass
class ModelOutput:
    logits:torch.Tensor
    pred_boxes:torch.Tensor

@torch.no_grad()
def compute_metrics(evaludation_results,image_processor,threshold=0.5,id2label=None):
    predictions,targets = evaludation_results.predictions,evaludation_results.label_ids
    #  - For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"
    image_sizes=[]
    post_processed_targets=[]
    post_processed_predictions=[]

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array(x['orig_size'] for x in batch))
        image_sizes.append(batch_image_sizes)

        for image_target in batch:
            boxes = torch.tensor(image_target['boxes'])
            boxes = convert_bbox_yolo_to_pascal(boxes,image_target['orig_size'])
            labels = torch.tensor(image_target['class_labels'])
            post_processed_targets.append({'boxes':boxes,'labels':labels})

    for batch,target_sizes in zip(predictions,image_sizes):
        batch_logits,batch_boxes = batch[1],batch[2]
        # 0 is loss
        output = ModelOutput(logits=torch.tensor(batch_logits),pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output,threshold=threshold,target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # compute metrics
    metric = MeanAveragePrecision(box_format='xyxy',class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # replace list of per class metrics with separate metric for each class
    classes = metrics.pop('classes')
    map_per_class = metrics.pop('map_per_class')
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics














