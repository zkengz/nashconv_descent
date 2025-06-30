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

If you find this project helpful, please cite our paper: 

```bibtex
@InProceedings{nashconv_descent,
    author="Zeng, Zekeng
        and Zhang, Youzhi
        and Yang, Peipei
        and Zhang, Mingyi
        and Zhang, Junge",
    editor="Mahmud, Mufti
        and Doborjeh, Maryam
        and Wong, Kevin
        and Leung, Andrew Chi Sing
        and Doborjeh, Zohreh
        and Tanveer, M.",
    title="Computing Approximate Nash Equilibrium in Two-Team Zero-Sum Games by NashConv Descent",
    booktitle="Neural Information Processing",
    year="2025",
    publisher="Springer Nature Singapore",
    address="Singapore",
    pages="167--181",
    abstract="Artificial intelligence algorithms have achieved superhuman performances in two-player zero-sum (2p0s) games by approximating Nash equilibrium. However, many real-world competitive scenarios are modeled as two-team zero-sum (2t0s) games, where a team of multiple players cooperatively competes against the other team. Despite the ubiquity, existing methods can only approximate Nash equilibrium in limited settings of 2t0s games. In this paper, we present an iterative algorithm to approximate Nash equilibrium in extensive-form general 2t0s games. To this end, we extend the concept of NashConv from 2p0s games to 2t0s games, which represents the total potential improvement if each player individually switched to its best response. NashConv provides a metric to measure the distance of a policy profile from a Nash equilibrium. The proposed algorithm, NashConv Descent, implements direct policy optimization based on NashConv. Utilizing tabular policy and iteratively performing policy gradient descent on NashConv for each player, our algorithm locally minimizes NashConv, thereby approximating Nash equilibrium. We evaluate our method on imperfect information benchmarks, such as multi-player Kuhn Poker and Leduc Poker. To the best of our knowledge, NashConv Descent is the first algorithm that empirically achieves approximate Nash equilibrium in extensive-form 2t0s games. Furthermore, our method attains the lowest NashConv compared to adaptations of existing equilibrium-computing algorithms tailored for extensive-form 2t0s games.",
    isbn="978-981-96-6585-3"
}
```

# Acknowledgement

The code is built on https://github.com/tansey/pycfr. We greatly appreciate Wesley Tansey's work.

# Contact

If you have any question about this repo or paper, feel free to leave an issue or email zengzekeng2022@ia.ac.cn.
