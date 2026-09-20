# Quantum Encoding SVM on UCI Spambase

Final Run 2 Experiment repository.

## Design
* **Dataset**: UCI Spambase (4,601 records, 57 features)
* **Preprocessing**: Stratified 80/20 split, StandardScaler (fit on train), PCA(16) (fit on train).
* **Branches**: Classical RBF-SVM, Basis Encoding, Angle Encoding, Amplitude Encoding, Phase Encoding.
* **Quantum Kernel**: $K(x,y) = |\langle \phi(x)|\phi(y) angle|^2$ via deterministic simulation.
* **SVM**: Scikit-Learn SVC, tuned on training data only.

Outputs are generated in `results/`.
