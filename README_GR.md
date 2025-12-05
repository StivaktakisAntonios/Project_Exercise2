# Neural LSH - Προσεγγιστική Αναζήτηση Πλησιέστερων Γειτόνων

**Έργο:** Κ23γ – 2η Προγραμματιστική Εργασία  
**Μάθημα:** Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα  
**Εξάμηνο:** Χειμερινό 2025-26

**Ομάδα 15:**  
- **Ανέστης Θεοδωρίδης** – ΑΜ: 1115201500212 – Email: sdi1500212@di.uoa.gr
- **Αντώνιος-Ραφαήλ Στιβακτάκης** – ΑΜ: 1115202200258 – Email: sdi2200258@di.uoa.gr

---

## Περιγραφή

Υλοποίηση του αλγορίθμου Neural LSH (Locality-Sensitive Hashing) για προσεγγιστική αναζήτηση πλησιέστερων γειτόνων χρησιμοποιώντας διαμέριση βασισμένη σε νευρωνικά δίκτυα.

Ο αλγόριθμος συνδυάζει:
- Κατασκευή γράφου k-NN με χρήση IVFFlat (C++ από την 1η Εργασία)
- Ισοκατανεμημένη διαμέριση γράφου με χρήση KaHIP
- Εκπαίδευση ταξινομητή MLP (Multi-Layer Perceptron) για πρόβλεψη διαμερισμάτων
- Στρατηγική multi-probe search για αποδοτική αναζήτηση

Η υλοποίηση ακολουθεί τις προδιαγραφές της 2ης Εργασίας (Κ23γ) και παράγει έξοδο συμβατή με τις μεθόδους της 1ης Εργασίας (LSH, Hypercube, IVFFlat, IVFPQ) για άμεση σύγκριση απόδοσης.

**Κατασκευή Γράφου k-NN:** Χρησιμοποιεί βελτιστοποιημένη υλοποίηση IVFFlat σε C++ από την 1η Εργασία για γρήγορη προσεγγιστική κατασκευή γράφου k-NN τόσο για MNIST όσο και για SIFT. Οι γράφοι k-NN προϋπολογίζονται και αποθηκεύονται για επαναχρησιμοποίηση σε πολλαπλά πειράματα.

## Δομή Έργου

```
Project_Exercise2/
├── Exercise/
│   ├── Modules/              # Βασικές ενότητες υλοποίησης
│   │   ├── config.py         # Ρυθμίσεις και σταθερές
│   │   ├── dataset_parser.py # Φόρτωση δεδομένων (MNIST, SIFT)
│   │   ├── graph_utils.py    # Κατασκευή γράφου k-NN
│   │   ├── partitioner.py    # Διαμέριση γράφου με KaHIP
│   │   ├── models.py         # Εκπαίδευση ταξινομητή MLP
│   │   ├── index_io.py       # Αποθήκευση/φόρτωση ευρετηρίου
│   │   ├── search.py         # Αλγόριθμοι αναζήτησης
│   │   └── Models/           # Υλοποιήσεις C++ από την 1η Εργασία
│   │       ├── build_knn_mnist.cpp  # Κατασκευή k-NN για MNIST
│   │       ├── build_knn_sift.cpp   # Κατασκευή k-NN για SIFT
│   │       ├── IVFFlat/      # Υλοποίηση IVF-Flat
│   │       ├── LSH/          # Υλοποίηση LSH
│   │       ├── Hypercube/    # Υλοποίηση Hypercube
│   │       ├── IVFPQ/        # Υλοποίηση IVF-PQ
│   │       └── Template/     # Εργαλεία (L2, data I/O)
│   ├── NeuralLSH/            # Εργαλεία γραμμής εντολών
│   │   ├── nlsh_build.py     # Script κατασκευής ευρετηρίου
│   │   └── nlsh_search.py    # Script αναζήτησης ερωτημάτων
│   ├── build_knn_mnist       # Compiled εκτελέσιμο k-NN για MNIST
│   └── build_knn_sift        # Compiled εκτελέσιμο k-NN για SIFT
├── experiments/              # Πειραματική επικύρωση
│   ├── run_experiments.py    # Αυτοματοποιημένη εκτέλεση πειραμάτων
│   └── results/              # Αποτελέσματα πειραμάτων
├── Raw_Data/                 # Σύνολα δεδομένων εισόδου
│   ├── MNIST/
│   └── SIFT/
├── build_knn_executables.sh  # Μεταγλώττιση k-NN executables
└── requirements.txt          # Εξαρτήσεις Python
```

## Εγκατάσταση

### Προαπαιτούμενα

- Python 3.10 ή νεότερο
- Εργαλείο διαμέρισης γράφων KaHIP
- Μεταγλωττιστής C++ με υποστήριξη C++17 (συνιστάται g++)

### Οδηγίες Εγκατάστασης

1. Κλωνοποίηση του αποθετηρίου:
```bash
git clone https://github.com/StivaktakisAntonios/Project_Exercise2.git
cd Project_Exercise2
```

2. Δημιουργία και ενεργοποίηση εικονικού περιβάλλοντος:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Σε Linux/Mac
```

3. Εγκατάσταση εξαρτήσεων Python:
```bash
pip install -r requirements.txt
```

4. Εγκατάσταση KaHIP:
```bash
# Σε Ubuntu/Debian
sudo apt-get install kahip

# Ή μεταγλώττιση από τον πηγαίο κώδικα
git clone https://github.com/KaHIP/KaHIP.git
cd KaHIP
./compile_withcmake.sh
export PATH=$PATH:$(pwd)/deploy
```

5. Μεταγλώττιση εκτελέσιμων C++ για k-NN γράφους:
```bash
./build_knn_executables.sh
```

Αυτό θα μεταγλωττίσει:
- `Exercise/build_knn_mnist` - Κατασκευή γράφου k-NN για MNIST με χρήση IVFFlat
- `Exercise/build_knn_sift` - Κατασκευή γράφου k-NN για SIFT με χρήση IVFFlat

6. Επαλήθευση εγκατάστασης:
```bash
which kaffpa
python -c "import torch; print(torch.__version__)"
./Exercise/build_knn_mnist -h
./Exercise/build_knn_sift -h
```

## Χρήση

### Κατασκευή Ευρετηρίου

Δημιουργία ευρετηρίου Neural LSH από ένα σύνολο δεδομένων:

```bash
python Exercise/NeuralLSH/nlsh_build.py \
    -d Raw_Data/MNIST/input.idx3-ubyte \
    -i indices/mnist_index \
    -type mnist \
    --knn 10 \
    -m 100 \
    --epochs 10
```

**Υποχρεωτικές Παράμετροι:**
- `-d, --dataset`: Διαδρομή αρχείου συνόλου δεδομένων
- `-i, --index`: Διαδρομή καταλόγου εξόδου ευρετηρίου
- `-type`: Τύπος δεδομένων (`mnist` ή `sift`)

**Προαιρετικές Παράμετροι:**
- `--knn`: Αριθμός πλησιέστερων γειτόνων για κατασκευή γράφου (προεπιλογή: 10)
- `-m, --partitions`: Αριθμός διαμερισμάτων/bins (προεπιλογή: 100)
- `--imbalance`: Ανοχή ανισορροπίας KaHIP (προεπιλογή: 0.03)
- `--kahip_mode`: Λειτουργία KaHIP: 0=FAST, 1=ECO, 2=STRONG (προεπιλογή: 2)
- `--layers`: Αριθμός κρυφών επιπέδων MLP (προεπιλογή: 3)
- `--nodes`: Πλάτος κρυφού επιπέδου MLP (προεπιλογή: 64)
- `--epochs`: Περίοδοι εκπαίδευσης (προεπιλογή: 10)
- `--batch_size`: Μέγεθος δέσμης εκπαίδευσης (προεπιλογή: 128)
- `--lr`: Ρυθμός εκμάθησης (προεπιλογή: 0.001)
- `--seed`: Σπόρος τυχαίοτητας για αναπαραγωγιμότητα (προεπιλογή: 1)

### Αναζήτηση σε Ευρετήριο

Αναζήτηση πλησιέστερων γειτόνων χρησιμοποιώντας κατασκευασμένο ευρετήριο:

```bash
python Exercise/NeuralLSH/nlsh_search.py \
    -d Raw_Data/MNIST/input.idx3-ubyte \
    -q Raw_Data/MNIST/query.idx3-ubyte \
    -i indices/mnist_index \
    -o outputs/mnist_output.txt \
    -type mnist \
    -N 1 \
    -T 5 \
    -range false
```

**Υποχρεωτικές Παράμετροι:**
- `-d, --dataset`: Διαδρομή αρχείου συνόλου δεδομένων
- `-q, --query`: Διαδρομή αρχείου ερωτημάτων
- `-i, --index`: Διαδρομή καταλόγου ευρετηρίου
- `-o, --output`: Διαδρομή αρχείου εξόδου αποτελεσμάτων
- `-type`: Τύπος δεδομένων (`mnist` ή `sift`)

**Προαιρετικές Παράμετροι:**
- `-N, --neighbors`: Αριθμός πλησιέστερων γειτόνων προς επιστροφή (προεπιλογή: 1)
- `-T, --top_bins`: Αριθμός κορυφαίων bins προς έλεγχο (προεπιλογή: 5)
- `-R, --radius`: Ακτίνα για R-near neighbors search (προεπιλογή: 2000 για MNIST, 2800 για SIFT)
- `-range, --range_search`: Ενεργοποίηση R-near neighbors: "true" ή "false" (προεπιλογή: true)
- `--max_queries`: Περιορισμός αριθμού ερωτημάτων (χρήσιμο για μεγάλα datasets όπως SIFT)

### Εκτέλεση Πειραμάτων

Εκτέλεση προκαθορισμένων πειραμάτων:

```bash
# Πείραμα MNIST
python experiments/run_experiments.py --dataset mnist

# Πείραμα SIFT
python experiments/run_experiments.py --dataset sift

# Όλα τα πειράματα
python experiments/run_experiments.py --all
```

## Αλγοριθμική Ροή

### Κατασκευή Ευρετηρίου

1. **Φόρτωση Συνόλου Δεδομένων**: Ανάγνωση δεδομένων εισόδου (μορφή MNIST ή SIFT)
2. **Κατασκευή Γράφου k-NN**: Δημιουργία προσεγγιστικού γράφου k-NN με χρήση IVFFlat σε C++
3. **Διαμέριση Γράφου**: Χρήση KaHIP για δημιουργία ισοκατανεμημένων διαμερισμάτων του γράφου k-NN
4. **Εκπαίδευση Ταξινομητή**: Εκπαίδευση MLP για πρόβλεψη αναθέσεων διαμερισμάτων από σημεία δεδομένων
5. **Αποθήκευση Ευρετηρίου**: Διατήρηση inverted index, εκπαιδευμένου μοντέλου και μεταδεδομένων

### Αναζήτηση Ερωτημάτων

1. **Φόρτωση Ευρετηρίου**: Φόρτωση inverted index και εκπαιδευμένου ταξινομητή
2. **Πρόβλεψη Διαμερισμάτων**: Χρήση MLP για πρόβλεψη των T πιο πιθανών bins για κάθε ερώτημα
3. **Multi-Probe Search**: Αναζήτηση υποψηφίων από επιλεγμένα bins
4. **Επανατοποθέτηση**: Υπολογισμός ακριβών αποστάσεων και επιστροφή των N πλησιέστερων γειτόνων

## Ρυθμίσεις

Βασικές παράμετροι στο `Exercise/Modules/config.py`:

```python
DEVICE = "cpu"          # Επιβολή εκτέλεσης μόνο σε CPU
RANDOM_SEED = 1         # Προεπιλεγμένος σπόρος τυχαιότητας
EPSILON = 1e-10         # Σταθερά αριθμητικής σταθερότητας
```

## Μορφές Συνόλων Δεδομένων

### MNIST
- Μορφή: IDX3-ubyte (δυαδική μορφή με 4-byte header)
- Διαστάσεις: 784 (28×28 pixels)
- Είσοδος: `Raw_Data/MNIST/input.idx3-ubyte`
- Ερωτήματα: `Raw_Data/MNIST/query.idx3-ubyte`

### SIFT
- Μορφή: fvecs (δυαδική μορφή με 4-byte πρόθεμα διάστασης)
- Διαστάσεις: 128
- Βάση: `Raw_Data/SIFT/sift_base.fvecs`
- Ερωτήματα: `Raw_Data/SIFT/sift_query.fvecs`
- Ground truth: `Raw_Data/SIFT/sift_groundtruth.ivecs`

## Μορφή Εξόδου

Τα αποτελέσματα αναζήτησης γράφονται σε μορφή συμβατή με την 1η Εργασία:

```
METHOD NAME: Neural LSH

Query: 0
Nearest neighbor-1: 12345
distanceApproximate: 123.45
distanceTrue: 123.45

Query: 1
Nearest neighbor-1: 67890
distanceApproximate: 234.56
distanceTrue: 234.56
...

Average AF: 1.0015
Recall@1: 0.9765
QPS: 117.63
tApproximateAverage: 0.008501
tTrueAverage: 0.077093
```

Κάθε ερώτημα περιλαμβάνει προσεγγιστικές και πραγματικές αποστάσεις, ακολουθούμενες από συγκεντρωτικές μετρικές.

## Πειραματικά Αποτελέσματα

Προϋπολογισμένα πειραματικά αποτελέσματα διαθέσιμα στο `outputs/`:
- `mnist_fast_N1_T5.txt` - MNIST με KaHIP FAST mode
- `mnist_eco_N1_T5.txt` - MNIST με KaHIP ECO mode
- `mnist_strong_N1_T5.txt` - MNIST με KaHIP STRONG mode
- `sift_fast_N1_T5.txt` - SIFT (100 ερωτήματα) με KaHIP FAST mode
- `sift_eco_N1_T5.txt` - SIFT (100 ερωτήματα) με KaHIP ECO mode
- `sift_strong_N1_T5.txt` - SIFT (100 ερωτήματα) με KaHIP STRONG mode
- `results_summary.txt` - Ολοκληρωμένη σύγκριση και ανάλυση

**Κύρια Αποτελέσματα:**
- MNIST ECO: 97.65% recall@1, 1.0015 AF, 117.63 QPS
- MNIST FAST: 96.62% recall@1, 1.0021 AF, 113.86 QPS
- MNIST STRONG: 96.85% recall@1, 1.0020 AF, 105.09 QPS
- SIFT ECO: 95.00% recall@1, 1.0011 AF (1M σημεία, ECO mode βέλτιστο)
- SIFT FAST: 86.00% recall@1, 1.0251 AF (1M σημεία)
- SIFT STRONG: 93.00% recall@1, 1.0065 AF (1M σημεία)

Το ECO mode δείχνει +9% βελτίωση recall για μεγάλης κλίμακας datasets (SIFT 1M).

## Βελτιστοποίηση Απόδοσης

### Κατασκευή Ευρετηρίου
- **Περισσότερα διαμερίσματα** (`-m`): Καλύτερη επιλεκτικότητα αλλά περισσότερη αποθήκευση
- **Υψηλότερο k** (`--knn`): Καλύτερη ποιότητα γράφου αλλά πιο αργή κατασκευή
- **Περισσότερες περίοδοι**: Καλύτερος ταξινομητής αλλά μεγαλύτερος χρόνος εκπαίδευσης
- **Ισχυρότερη λειτουργία KaHIP**: Καλύτερα διαμερίσματα αλλά πιο αργή διαμέριση

### Αναζήτηση Ερωτημάτων
- **Περισσότερα bins** (`-T`): Υψηλότερο recall αλλά πιο αργή αναζήτηση
- **Μικρότερη ακτίνα** (`-R`): Αυστηρότερο φιλτράρισμα R-near neighbors (όταν range=true)
- **Περιορισμός ερωτημάτων** (`--max_queries`): Ταχύτερη αξιολόγηση για μεγάλα datasets όπως SIFT

## Σημειώσεις Ανάπτυξης

- **Μόνο CPU**: Η υλοποίηση χρησιμοποιεί PyTorch μόνο για CPU (δεν απαιτείται GPU)
- **Ντετερμινιστική**: Σταθεροί σπόροι τυχαιότητας εξασφαλίζουν αναπαραγώγιμα αποτελέσματα
- **Αποδοτική μνήμη**: Επεξεργασία σε δέσμες για μεγάλα σύνολα δεδομένων
- **Αρθρωτός σχεδιασμός**: Σαφής διαχωρισμός μεταξύ modules για συντηρησιμότητα
- **Συμβατότητα με Εργασία 1**: Η μορφή εξόδου ταιριάζει με την Εργασία 1 για άμεση σύγκριση με LSH, Hypercube, IVFFlat και IVFPQ

## Πειραματική Σύγκριση

Για την πειραματική αναφορά σύγκρισης του Neural LSH με τις μεθόδους της Εργασίας 1:

1. **Εκτέλεση πειραμάτων Neural LSH**:
   ```bash
   python experiments/run_experiments.py --all
   ```

2. **Σύγκριση μετρικών** (από αμφότερες τις εργασίες):
   - Recall@N: Κλάσμα αληθινών γειτόνων που βρέθηκαν
   - Average AF: Συντελεστής προσέγγισης (λόγος απόστασης)
   - QPS: Ερωτήματα ανά δευτερόλεπτο (throughput)
   - tApproximate: Μέσος χρόνος ερωτήματος
   - tTrue: Μέσος χρόνος υπολογισμού ground truth

3. **Βελτιστοποίηση υπερπαραμέτρων**: Χρήση `experiments/configs/` για δοκιμή διαφορετικών ρυθμίσεων:
   - Αριθμός διαμερισμάτων (`m`)
   - Μέγεθος γράφου k-NN (`k`)
   - Αρχιτεκτονική MLP (`layers`, `nodes`)
   - Παράμετροι εκπαίδευσης (`epochs`, `batch_size`, `lr`)
   - Βάθος multi-probe (`T`)

Τα αποτελέσματα θα αποθηκευτούν στο `experiments/results/` ως αρχεία JSON για ανάλυση.

## Αντιμετώπιση Προβλημάτων

### Δεν βρέθηκε το KaHIP
```bash
which kaffpa
export PATH=$PATH:/path/to/KaHIP/deploy
```

### Ανεπαρκής μνήμη
- Μείωση μεγέθους δέσμης: `--batch_size 64`
- Μείωση διαμερισμάτων: `-m 50`
- Χρήση μικρότερου k: `--knn 5`

### Χαμηλό recall
- Αύξηση ελεγχόμενων bins: `-T 10`
- Αύξηση υποψηφίων επανατοποθέτησης: `-R 100`
- Χρήση περισσότερων διαμερισμάτων: `-m 200`
- Μεγαλύτερη εκπαίδευση: `--epochs 20`

## Αναφορές

- KaHIP: https://github.com/KaHIP/KaHIP
- PyTorch: https://pytorch.org/
