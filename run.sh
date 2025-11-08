python DPE-MVS/colmap2mvsnet.py --dense_folder ../colmap_anno/lane/walking1 --save_folder output
echo "Finished colmap2mvsnet.py"
./build/DPE output 0 --no_viz --no_fusion --no_weak