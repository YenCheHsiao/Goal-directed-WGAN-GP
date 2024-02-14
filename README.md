# Goal-directed-Wasserstein Generative Adversarial Networks (WGAN) with Gradient penalty(GP)
* The manuscript for the code is submitted to a journal for potential publication.
* The preprint of the manuscript can be found at:
  * Yen-Che Hsiao, Abhishek Dutta. Epitope Generation for Peptide-based Cancer Vaccine using Goal-directed Wasserstein Generative Adversarial Network with Gradient Penalty, 07 June 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3015960/v1]
* To reproduce the figures or to retrain our model, please see the run file in the code folder.
## Instruction
* The main algorithm is in the 'Training/Goal-directed_WGAN-GP_retrain.py' line 701-720.
  
* To generate the figures in the paper, please create a folder called results under Goal-directed-WGAN-GP
 and go to the folder code/Figures in paper/ and run
  * "5. Number of immunogenic epitopes.py" for Fig. 2. (b)
  * "6. IC50_compare.py" for Fig. 2. (d) and (e)
  * "7. Load_My similarity 3.py" for Fig. 2. (c)
  * "8. load learning data.py" for Fig. 2. (f)
  * "9. dot_plot.py" for Fig. 2. (g), (h), (i), (j), and (k)
  * The generated figures will be stored in Goal-directed-WGAN-GP/results
    
* Example for training our designed Goal-directed_WGAN-GP and show the predicted immunogenicity score of the generated peptides.
  1. Create the folder "Goal-directed-WGAN-GP/results/Goal-directed_WGAN-GP/epoch1000"
  2. Run "Goal-directed-WGAN-GP/code/Training/(epoch 1000) Goal-directed_WGAN-GP_retrain.py" 
  3. Run "Reproduced figures/1. GAN+CNN (no label)-generator data for scoring.py" to generate 10000 peptides from the Goal-directed-WGAN-GP
  4. Run "Reproduced figures/2. Remove_placeholder_morethan2.py" to remove peptide sequences with placeholder more than 2
  5. Run "Reproduced figures/3. GAN+CNN (no label)-scoring from file.py" to get the output file "deepimmuno-Goal-directed_WGAN-GP-bladder_cancer-scored_epoch1000-batch10000.txt" in 'results/'
  * The output file "deepimmuno-GANRL-bladder_cancer-scored_epoch1000-batch10000.txt" in 'results/Goal-directed_WGAN-GP/ep1000/' will have the 10000 generated peptides with their binding HLA and their predicted immunogenicity from DeepImmuno-CNN.

* If you are interested in retraining the generator and getting new results, please run the codes below.
  * Retrain the Goal-directed WGAN-GP with 0,20,40,60,80,100, and 1000 epochs
    1. Create the folders with name "epoch0" , "epoch20", "epoch40", "epoch60", "epoch80", "epoch100", and "epoch1000" under "Goal-directed-WGAN-GP/results/Goal-directed_WGAN-GP/".
    2. Run "Goal-directed-WGAN-GP/code/Training/(epoch 0) Goal-directed_WGAN-GP_retrain.py"
    3. Run "Goal-directed-WGAN-GP/code/Training/(epoch 20) Goal-directed_WGAN-GP_retrain.py"
    4. Run "Goal-directed-WGAN-GP/code/Training/(epoch 40) Goal-directed_WGAN-GP_retrain.py"
    5. Run "Goal-directed-WGAN-GP/code/Training/(epoch 60) Goal-directed_WGAN-GP_retrain.py"
    6. Run "Goal-directed-WGAN-GP/code/Training/(epoch 80) Goal-directed_WGAN-GP_retrain.py"
    7. Run "Goal-directed-WGAN-GP/code/Training/(epoch 100) Goal-directed_WGAN-GP_retrain.py"
    8. Run "Goal-directed-WGAN-GP/code/Training/(epoch 1000) Goal-directed_WGAN-GP_retrain.py"
  * Retrain the vanilla WGAN-GP with 1000 epochs
    1. Create the folder "Goal-directed-WGAN-GP/results/WGAN-GP/epoch1000".
    2. Run "Goal-directed-WGAN-GP/code/Training/WGAN-GP_retrain.py"
  * Generate the figures except for the binding affinity (BA), since BA needs to input the generated peptides from the output in "2. (all) Remove_placeholder_morethan2.py" (..._rmv.txt) to NetMHCpan - 4.1 at https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/, download the predicted BA, save the file into a CSV file, run IC50_compare.py to get the comparison.
    1. Run "Goal-directed-WGAN-GP/code/Reproduced figures/1. (all) GAN+CNN (no label)-generator data for scoring.py"
    2. Run "Goal-directed-WGAN-GP/code/Reproduced figures/2. (all) Remove_placeholder_morethan2.py"
    3. Run "Goal-directed-WGAN-GP/code/Reproduced figures/3. (all) GAN+CNN (no label)-scoring from file.py"
    4. Run "Goal-directed-WGAN-GP/code/Reproduced figures/4. (all) My similarity.py"
    5. Run "Goal-directed-WGAN-GP/code/Reproduced figures/5. (all) Number of immunogenic epitopes.py"
    6. Run "Goal-directed-WGAN-GP/code/Reproduced figures/7. Load_My similarity 3.py"
    7. Run "Goal-directed-WGAN-GP/code/Reproduced figures/8. load learning data.py"
    8. Run "Goal-directed-WGAN-GP/code/Reproduced figures/9. dot_plot.py"
## Acknowledges
* The training data of the DeepImmuno CNN, WGAN-GP and some part of the scripts are from the work in: https://github.com/frankligy/DeepImmuno
*DeepImmuno: deep learning-empowered prediction and generation of immunogenic peptides for T-cell immunity*, Briefings in Bioinformatics, May 03 2021 (https://doi.org/10.1093/bib/bbab160)
* The training dataset for the GANs is from:
Wu, Jingcheng, et al. "TSNAdb: a database for tumor-specific neoantigens from immunogenomics data analysis." Genomics, proteomics & bioinformatics 16.4 (2018): 276-282.
