cat ../data/experiment_scripts/e_losses.sh | parallel -u -j 4
cat ../data/experiment_scripts/e_distance_bed.sh | parallel -u -j 10
cat ../data/experiment_scripts/e_distance_optimal.sh | parallel -u -j 3
cat ../data/experiment_scripts/e_distance_tree.sh | parallel -u -j 10
python error_landscape_smoothness.py -calculate_distance -baseline edit
python error_landscape_smoothness.py -precompute_ranks
cat ../data/experiment_scripts/e_plots_smoothness.sh | parallel -u -j 5