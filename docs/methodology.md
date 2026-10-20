# Methodology and limitations

Uses the offline sklearn digits dataset. A training/validation/test split isolates kernel gamma selection from test reconstruction scores. Scaling is fitted only on training pixels. Kernel PCA is compared with ordinary PCA using two components and reconstruction in standardized feature space. Reconstructed pixels are inverse-scaled only for visualization. Class labels do not enter the compression transform.
