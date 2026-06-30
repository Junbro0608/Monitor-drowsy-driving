FINAL_3_REPORT
├─ 01_eye_state_comparison
│  ├─ models
│  │  ├─ mobilenetv3_small_best_by_f1.pt / .onnx
│  │  └─ yolov8n_cls_best_by_f1.pt / .onnx
│  └─ notebooks
│     ├─ 01_mobilenetv3_optuna_train_colab.ipynb
│     └─ 02_yolov8n_cls_optuna_train_colab.ipynb
├─ 02_eye_roll_aug_finetune
│  ├─ models
│  │  └─ roll_aug_yolov8n_cls_finetuned_best.pt / .onnx
│  └─ training_code
│     └─ run_yolo_roll_aug_dataset_finetune.py
├─ 03_benchmark
│  └─ code
│     └─ yolo_trt_roll_aug_export_eval.py
└─ 04_eye_position_detection
   ├─ models
   │  └─ eye_position_stage2_finetuned_best.pt / .onnx
   └─ training_code
      ├─ 01_train_initial_eye_position_detector.py
      └─ 02_finetune_eye_position_detector.py