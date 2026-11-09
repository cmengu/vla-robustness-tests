G=(h100-47 a100-40); i=0
for S in libero_spatial libero_object libero_goal libero_10; do for H in a b; do
  sbatch -J a-$S-$H --gpus=${G[i%2]} ~/t3a.sh $S $H; i=$((i+1))
done; done
squeue --me -o "%i %j %T"
