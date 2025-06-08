# Feature Vector Transformer (FeVeT)
Data, code and results accompanying the Feature Vector Transformer (FeVeT) paper.

The repository is structured as follows:
- [code](/code/): all scripts for data preparation, the model, model training and evaluation
- [data](/data/): CLTS (List et al., 2024) data, data for the proto-language reconstruction task (List et al., 2022a) and data for the reflex prediction task (List et al., 2022b)
- [error_analysis](/error_analysis/): output files of the error analysis
- [model_checkpoints](/model_checkpoints/): the trained models
- [results](/results/): detailed and aggregated model performances
  - [predictions](/results/predictions/): tables with model outputs vs. gold forms
- [metrics](/metrics/): functions to calculate the metrics

If you use this code in your work, please cite:
> Wientzek, T. (forthcoming). Using feature vectors for automated phonological reconstruction and reflex prediction. Open Research Europe.

## How to run
To replicate the findings of the study, preferably create a fresh Virtual Environment (tested with Python 3.12, CUDA 12.8) after cloning this folder.  
This can be done by navigating to this folder in the command prompt and running the following command:  
```bash
python -m venv venv
```  

Then, to activate the Virtual Environment:  
Mac/Linux:  
```bash
source venv/bin/activate
```
Windows:  
```bash
venv\Scripts\activate  
```
To be safe, upgrade pip and then install all required packages via  
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```


All experiments can be run using the run.py script, with the option to tweak many parameters.  
If you would for example like to run the model on the reflex prediction task using 16 epochs in training and fine-tuning, only using test proportions of 30% and 50% and a dropout of 0.3 in the encoder and decoder, use:  
```bash
python run.py reflex n_epochs=16 fine_tune_epochs=16 missing_prob=0.50,0.30 dropout=0.3
```
All results and predictions are exported to the [results](/results/) folder.  


To see all tweakable parameters, run:  
```bash
python run.py -h
``` 


## References
List, J.-M., Anderson, C., Tresoldi, T., Rzymski, C. & Forkel, R.. (2024). CLTS. Cross-Linguistic Transcription Systems (v2.3.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10997741

List, J.-M., Forkel, R., & Hill, N. (2022a). A New Framework for Fast Automated Phonological Reconstruction Using Trimmed Alignments and Sound Correspondence Patterns. Proceedings of the 3rd Workshop on Computational Approaches to Historical Language Change, 89–96. https://doi.org/10.18653/v1/2022.lchange-1.9

List, J.-M., Vylomova, E., Forkel, R., Hill, N., & Cotterell, R. (2022b). The SIGTYP 2022 Shared Task on the Prediction of Cognate Reflexes. In E. Vylomova, E. Ponti, & R. Cotterell (Eds.), Proceedings of the 4th Workshop on Research in Computational Linguistic Typology and Multilingual NLP (S. 52–62). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.sigtyp-1.7

