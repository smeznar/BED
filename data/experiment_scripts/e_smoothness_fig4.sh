cat ../data/experiment_scripts/e_losses.sh | parallel -u -j 4
cat ../data/experiment_scripts/e_distance_optimal.sh | parallel -u -j 3
cat ../data/experiment_scripts/e_distance_tree.sh | parallel -u -j 10
cat ../data/experiment_scripts/e_distance_bed.sh | parallel -u -j 10