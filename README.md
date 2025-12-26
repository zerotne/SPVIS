# SPVIS: Stabilizing Feature Propagation for Video Instance Segmentation 

## Introduction
Video instance segmentation (VIS) extends instance-level understanding from static images to continuous video, demanding accurate pixel-level masks and consistent identity association across frames. Feature-propagation approaches have become prominent for their efficiency, yet they remain fundamentally constrained by error accumulation over time and feature degradation during propagation. We present SPVIS, a feature propagation-based VIS framework with in-memory object-query propagation that explicitly targets these two bottlenecks while preserving computational efficiency. The design comprises two complementary parts: a Progressive Tracker (PGT) that performs cross-clip association with filtering to curb temporal error accumulation, and joint feature-preserving modeling-instantiated by the Refinement Compensator (RCP) and the Spatial Interaction Module (SIM)-that couples temporal compensation with discriminative, mask-guided spatial interaction to maintain high-quality object queries during propagation. Across standard benchmarks, SPVIS attains 69.5, 64.6, 51.9, and 54.3 AP on YouTube-VIS 2019, 2021, 2022, and OVIS, respectively, delivering competitive results with favorable accuracy-efficiency trade-offs in online and offline settings. The proposed formulation provides a targeted and lightweight path to association over long sequences, including scenarios with low frame rates and occlusions.




