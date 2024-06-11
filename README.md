# Multi-modal Arousal and Valence Estimation under Noisy Conditions 

As emotions play a central role in human communication, automatic emotion recognition has attracted increasing attention in the last two decades. While multimodal systems enjoy high performances on lab-controlled data, they are still far from providing ecological validity on non-lab-controlled, namely ‘in-the-wild’ data. This work investigates audiovisual deep learning approaches for emotion recognition in-the-wild problem. We particularly explore the effectiveness of architectures based on fine-tuned Convolutional Neural Networks (CNN) and Public Dimensional Emotion Model (PDEM), for video and audio modality, respectively. We compare alternative temporal modeling and fusion strategies using the embeddings from these multi-stage trained modality-specific Deep Neural Networks (DNN). We report results on the AffWild2 dataset under Affective Behavior Analysis in-the-Wild 2024 (ABAW’24) challenge protocol

<h2 align="center"> The pipeline of the developed ER system 
</h2>
<p align="center">
<img src="https://github.com/ABAWVAEXPR/ABAW2024/blob/main/figures/pipeline.png" alt="The pipeline of the developed ER system"/>
</p>
<h4> (a) the Audio-Transformer-based dynamic emotion recognition system, (b) the Visual-Transformer-based dynamic emotion recognition system, and (c) -- the Kernel ELM-based emotion recognition system. W -- the temporal window size (in the number of frames), N -- the number of neurons in the decision-making head (2 for regression task). </h4>
&NewLine;
&NewLine;


Model weights are available at [Google Drive](https://drive.google.com/drive/folders/12LLx9DiEJSlnzgL745m9z_XAz1Rw_Vz6?usp=sharing).
