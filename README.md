This repository contains the code for the paper *Computing Approximate Nash Equilibrium in Two-team Zero-sum Games by NashConv Descent*. 

# Usage

1. Install packages (some may be unnecessary). 

    ```bash
    pip install -r requirements.txt
    ```

2. Run NashConv Descent and baseline algorithms.

    ```bash
    python run_kuhn_2v1.py
    python run_leduc_2v1.py
    python run_kuhn_2v2.py
    ```

3. Plot the results. Note the results from the paper are already stored in `/results`.  

    ```bash
    python plot.py
    ```
    
# Citation

Citation information for this project will be added once the related paper is published.

# Acknowledgement

The code is built on https://github.com/tansey/pycfr. We have great appreciation for Wesley Tansey's work.

# Contact

If you have any question about this repo or paper, feel free to leave an issue or email zengzekeng2022@ia.ac.cn.
