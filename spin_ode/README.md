```sh
python spin_ode/traj_fit.py -c spin_ode/toy.yaml -t traj_fit -s experiments/traj_fit

python spin_ode/est_rate_dydt.py -c spin_ode/toy.yaml -t est_rate_neural_ode -s experiments/est_rate_nn/
python spin_ode/est_rate_dydt.py -c spin_ode/toy.yaml -t est_rate_finite_diff -s experiments/est_rate_finite_diff/

python spin_ode/est_rate_traj.py -c spin_ode/toy.yaml -t est_rate_traj -s experiments/est_rate_traj/
```