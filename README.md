**Aperion AI: FlashCost Agent (Local Mirror)** \
**Document Version:** 1.0 \
**Principal Lead:** Abhishek Prakash \
**Project: ** FlashCost Manufacturing Cost Estimation Agent \
**Executive Summary**
This repository serves as the architectural blueprint and local implementation for the FlashCost Agent. It bridges the gap between the initial CNN model development (V1–V3) and the current modular Reasoning Engine, which uses vector-based retrieval of historical evidence to justify cost predictions.
________________

**Action Required: Hand-off Artifacts**
Due to GitHub’s 100MB file size limit, the following heavy artifacts are excluded from this repository. You must download them from
https://github.com/abp15/aperion_local_agent/releases/tag/v1.0.0-local-mirror 
and place them in the following paths:
Artifact
	Type
	Destination Path
	cnn_image.tar
	Docker Binary
	./ (Root)
	vector_image.tar
	Docker Binary
	./ (Root)
	variables/
	Model Weights
	cnn_service/my_model/variables/
	vector_search_index_data.json
	Vector Data
	vector_service/data/
	________________


**Quick Start Instructions**
Load Docker Containers:
Bash
docker load -i cnn_image.tar
docker load -i vector_image.tar
1. 2. Restore Large Assets:
Ensure the variables/ folder and the .json index are moved to the paths specified in the table above.
Orchestrate Services:
Bash
docker-compose up -d
   3. Launch the Dashboard:
Bash
pip install -r requirements.txt
streamlit run app.py
   4. ________________

**Part I: The CNN Development Playbook (V1–V3)**
Lessons learned while building the "Brain" (Cost Prediction Model).
1. Data & GCS Strategy
   * Goal: Generate representative datasets of sketches and manufacturing costs.
   * Key Learning: Ensure simulated file paths in labels.csv match the directory structure intended for cloud deployment.
   * Regional Affinity: Maintain buckets (e.g., apparel_staging_central) in us-central1 to minimize latency and egress costs.
2. "Robust" task.py Architecture
   * Serving Signature Parity: Training scripts must ensure production behavior matches training logic.
   * Internal Decoding: Use tf.io.decode_base64 followed by tf.io.decode_image inside the model’s Lambda layer.
   * Tensor-Safe Pathing: Use tf.strings.join for GCS paths to remain compatible with TensorFlow Tensors.
________________


**Part II: FlashCost Agent (Modular Architecture)**
The Reasoning Engine that provides "Evidence" for every prediction.
   1. Input Orchestration: Handles ingestion of sketches and returns raw image bytes as a unified input for CNN and Vector Search.
   2. Manufacturing Cost Prediction: Bypasses the 1.5MB JSON payload limit by sending images as URL-safe Base64 strings.
   3. Historical Evidence Discovery: * Featurizer: MobileNetV2 extracts a 1,280-dimension vector.
   * Vector Search: Finds the top 5 nearest neighbors in the 10,000-image training set.
   4. Data Fusion: Cross-references Index IDs with labels.csv to retrieve historical cost "Ground Truth."
________________


