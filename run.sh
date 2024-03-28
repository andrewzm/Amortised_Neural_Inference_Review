#!/bin/bash
unset R_HOME

set -e

echo ""
echo "##### Starting illustration of Bayes classifier of Section 4 #####"
echo ""

Rscript src/0_Bayes_classifier.R

echo ""
echo "##### Starting illustration study of Section 5 #####"
echo ""

# Generate data
Rscript src/1_Generate_GP_Data.R
Rscript src/1_Generate_invMSP_Data.R

# Run likelihood-based methods
Rscript src/2_GP_MCMC.R
Rscript src/2_MSP_MaxMix.R

# Run neural methods
for statmodel in GP 
do
    echo ""
    echo "##### Starting experiments for $statmodel model #####"
    echo ""
    Rscript src/3_fKL.R --statmodel=$statmodel
    Rscript src/4_rKL.R --statmodel=$statmodel
    
    Rscript src/4_rKL_MDN.R --statmodel=$statmodel
    Rscript src/5_rKL_Synthetic_Naive.R --statmodel=$statmodel
    Rscript src/6_rKL_Synthetic_MutualInf.R --statmodel=$statmodel
    
    Rscript src/7_NBE.R --statmodel=$statmodel
    python src/8_NRE_SBI.py 
    Rscript src/8_NRE_SBI.R 
done

for statmodel in MSP
do
    echo ""
    echo "##### Starting experiments for $statmodel model #####"
    echo ""
    Rscript src/3_fKL.R --statmodel=$statmodel
    Rscript src/8_NRE.R --statmodel=$statmodel
done


# Plot the results
Rscript src/9a_Plot_Synth_Lik.R 
Rscript src/9b_Plot_Scatter_plots.R
Rscript src/9c_Plot_Micro_Results.R
Rscript src/9d_Table_Results_Summary.R
Rscript src/9e_Plot_Micro_Results_MSP.R
Rscript src/9f_Plot_Scatter_plots_MSP.R

echo ""
echo "######## Everything finished! ############"
echo ""
