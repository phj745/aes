# for i in {0..4}
# do
#   echo "Running inference on fold $i"
#   python infer.py --input_dir "../../data/AES/fold$i/train.csv" --label
# done
for i in {0..4}
do
  echo "Running inference on fold $i"
  python infer.py --input_dir "../../data/AES/fold$i/val.csv" --label
done