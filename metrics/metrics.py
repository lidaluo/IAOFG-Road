from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support

from . import reference_topo_apls as _ref


class MetricsCalculator:
    """
    拓扑与 APLS 与仓库根 ``calculate_topo_metrics.py``、``standard_apls.py`` 对齐：

    - TOPO-P/R/F1：图栅格 → 骨架 → 随机采样 → 半径内 KDTree 匹配。
    - APLS：图栅格为 mask → sknw 骨架图 → SpaceNet 风格 ``1 - mean(penalty)``。
    """

    def __init__(
        self,
        max_gt_nodes_for_apls: int = 0,
        *,
        topo_sample_radius: float = 10.0,
        topo_n_samples: int = 8000,
        topo_sample_seed: int = 0,
        topo_raster_line_width: int = 3,
        apls_snap_threshold: float = 50.0,
        apls_seed: int = 42,
    ):
        """
        max_gt_nodes_for_apls
            若 >0：用作 ``apls_n_pairs``（与旧版「控制采样量」习惯兼容）；若为 0：使用 200（与 standard_apls 默认一致）。
        topo_sample_radius
            骨架点匹配半径（像素），对应 calculate_topo_metrics.py 的 ``--radius`` 默认 10。
        topo_n_samples
            骨架采样点数，对应 ``--n_samples`` 默认 8000。
        """
        self.max_gt_nodes_for_apls = int(max_gt_nodes_for_apls)
        self.topo_sample_radius = float(topo_sample_radius)
        self.topo_n_samples = int(topo_n_samples)
        self.topo_sample_seed = int(topo_sample_seed)
        self.topo_raster_line_width = int(topo_raster_line_width)
        self.apls_n_pairs = int(max_gt_nodes_for_apls) if int(max_gt_nodes_for_apls) > 0 else 200
        self.apls_snap_threshold = float(apls_snap_threshold)
        self.apls_seed = int(apls_seed)

    def calculate_iou(self, pred_mask, gt_mask):
        """计算IoU（交并比）"""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        return iou
    
    def calculate_f1_score(self, pred_mask, gt_mask):
        """计算F1-score"""
        true_positive = np.logical_and(pred_mask, gt_mask).sum()
        false_positive = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        false_negative = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1

    def calculate_apls(self, pred_graph: nx.Graph, gt_graph: nx.Graph, canvas_hw: tuple[int, int] | None = None):
        """
        与 ``standard_apls.compute_apls`` 一致：mask 由图栅格化得到，再 sknw 最短路惩罚。
        ``canvas_hw`` 为 ``(H, W)``；若为 None，则由节点 ``pos`` 推断（至少 2048）。
        """
        if len(gt_graph.nodes()) < 2:
            return 1.0 if len(pred_graph.nodes()) == len(gt_graph.nodes()) else 0.0
        hw = canvas_hw if canvas_hw is not None else _ref.infer_canvas_hw(pred_graph, gt_graph)
        try:
            return _ref.compute_apls_from_graphs(
                pred_graph,
                gt_graph,
                hw,
                n_pairs=self.apls_n_pairs,
                snap_threshold=self.apls_snap_threshold,
                seed=self.apls_seed,
                raster_line_width=self.topo_raster_line_width,
            )
        except ImportError:
            raise

    def calculate_topo_precision_recall_f1(
        self,
        pred_graph: nx.Graph,
        gt_graph: nx.Graph,
        canvas_hw: tuple,
        buffer_px: int = 5,
        line_width: int = 3,
    ) -> tuple[float, float, float]:
        """
        与 ``calculate_topo_metrics.py`` 的采样 TOPO 一致。

        ``buffer_px`` / ``line_width`` 为历史参数名；实际使用 ``topo_sample_radius`` 与 ``topo_raster_line_width``（构造器可配）。
        """
        _ = buffer_px
        lw = int(line_width) if line_width else self.topo_raster_line_width
        return _ref.topo_precision_recall_f1_from_graphs(
            pred_graph,
            gt_graph,
            (int(canvas_hw[0]), int(canvas_hw[1])),
            radius=self.topo_sample_radius,
            n_samples=self.topo_n_samples,
            seed=self.topo_sample_seed,
            raster_line_width=lw,
        )

    def calculate_topo_f1(
        self,
        pred_graph: nx.Graph,
        gt_graph: nx.Graph,
        canvas_hw: tuple,
        buffer_px: int = 5,
        line_width: int = 3,
    ) -> float:
        _, _, f1 = self.calculate_topo_precision_recall_f1(
            pred_graph, gt_graph, canvas_hw, buffer_px=buffer_px, line_width=line_width
        )
        return f1

    def calculate_topo_iou(self, pred_graph, gt_graph):
        """计算TopoIoU（基于图结构的交并比）"""
        # 提取边集合
        pred_edges = set(tuple(sorted(edge)) for edge in pred_graph.edges())
        gt_edges = set(tuple(sorted(edge)) for edge in gt_graph.edges())
        
        # 计算交集和并集
        intersection = len(pred_edges & gt_edges)
        union = len(pred_edges | gt_edges)
        
        topo_iou = intersection / union if union > 0 else 0
        return topo_iou
    
    def calculate_intersection_metrics(self, pred_intersections, gt_intersections, threshold=10):
        """计算交叉口检测的Precision、Recall、F1"""
        if len(pred_intersections) == 0 and len(gt_intersections) == 0:
            return 1.0, 1.0, 1.0
        
        # 标记已匹配的真实交叉口
        matched_gt = [False] * len(gt_intersections)
        true_positives = 0
        
        # 对每个预测的交叉口，找到最近的真实交叉口
        for pred in pred_intersections:
            min_distance = float('inf')
            closest_gt_idx = -1
            
            for i, gt in enumerate(gt_intersections):
                distance = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_gt_idx = i
            
            if min_distance < threshold and not matched_gt[closest_gt_idx]:
                true_positives += 1
                matched_gt[closest_gt_idx] = True
        
        precision = true_positives / len(pred_intersections) if len(pred_intersections) > 0 else 0
        recall = true_positives / len(gt_intersections) if len(gt_intersections) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_intersection_type_accuracy(self, pred_intersections, gt_intersections, pred_graph, gt_graph):
        """计算交叉口类型分类准确率"""
        # 这里简化处理，仅基于度数来分类交叉口类型
        # T字交叉口：度数为3
        # 十字交叉口：度数为4
        # 其他：度数为2或其他
        
        correct = 0
        total = 0
        
        # 对每个真实交叉口，找到对应的预测交叉口
        for gt_idx, gt_intersection in enumerate(gt_intersections):
            # 找到最近的预测交叉口
            min_distance = float('inf')
            closest_pred_idx = -1
            
            for pred_idx, pred_intersection in enumerate(pred_intersections):
                distance = np.sqrt((pred_intersection[0] - gt_intersection[0])**2 + (pred_intersection[1] - gt_intersection[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pred_idx = pred_idx
            
            if min_distance < 10:  # 阈值为10像素
                # 获取真实交叉口的度数
                gt_degree = gt_graph.degree(gt_idx)
                # 获取预测交叉口的度数
                if closest_pred_idx in pred_graph.nodes():
                    pred_degree = pred_graph.degree(closest_pred_idx)
                    # 判断类型是否一致
                    if (gt_degree == 3 and pred_degree == 3) or (gt_degree == 4 and pred_degree == 4):
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def calculate_all_metrics(self, pred_results, gt_results):
        """计算所有评价指标"""
        # 像素级指标
        pred_mask = pred_results['segmentation_mask']
        gt_mask = gt_results['segmentation_mask']
        iou = self.calculate_iou(pred_mask, gt_mask)
        f1_score = self.calculate_f1_score(pred_mask, gt_mask)
        
        # 拓扑级指标
        pred_graph = pred_results['graph']
        gt_graph = gt_results['graph']
        ch, cw = int(pred_mask.shape[0]), int(pred_mask.shape[1])
        apls = self.calculate_apls(pred_graph, gt_graph, canvas_hw=(ch, cw))
        topo_iou = self.calculate_topo_iou(pred_graph, gt_graph)
        
        # 交叉口级指标
        pred_intersections = pred_results['intersections']
        gt_intersections = gt_results['intersections']
        precision, recall, f1_intersection = self.calculate_intersection_metrics(pred_intersections, gt_intersections)
        intersection_type_accuracy = self.calculate_intersection_type_accuracy(pred_intersections, gt_intersections, pred_graph, gt_graph)
        
        return {
            'pixel_level': {
                'iou': iou,
                'f1_score': f1_score
            },
            'topology_level': {
                'apls': apls,
                'topo_iou': topo_iou
            },
            'intersection_level': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_intersection,
                'type_accuracy': intersection_type_accuracy
            }
        }
