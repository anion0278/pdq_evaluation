import subprocess, os, shutil, re 
import numpy as np
import read_files
import matplotlib.pyplot as plt

path_to_models = "E:/models/FINAL_third_party/RHD"
dataset_anns_path = "E:/datasets/real_merged_l515_640x480/instances_hands_full.json"
output_file_name = "pdq_score_threshold_evals.txt"
ap_evaluator_path = "D:/source/repos/anion0278/SOLO"
coco_dets_file_name = "out.pkl.bbox.json"
rvc_dets_file_name = "rvc1_det.json"

arch_translation = {
    "mask_rcnn_r50_fpn" : "Mask R-CNN ResNet50",
    "mask_rcnn_r101_fpn" : "Mask R-CNN ResNet101",
    "solov2_light_448_r50_fpn" : "SOLOv2 ResNet50",
    "solov2_r101_fpn" : "SOLOv2 ResNet101"
}

channels_translation = {
    1 : "Depth",
    3 : "RGB",
    4 : "RGB-D",
}

def list_all_dirs_only(path):
    return next(os.walk(path))[1]

def parse_config_and_channels_from_checkpoint_path(checkpoint_path):
    import re, os
    matches = re.search(r"^\d[A-Za-z0-9]+-(?P<arch>\w+)_(?P<channels>\d)ch", os.path.basename(checkpoint_path))
    return matches.group('arch'), int(matches.group('channels'))

def eval_pdq(model_dir):
    # generate predictions and evaluate mAP 
    # os.system(f"cd {ap_evaluator_path} && python {ap_evaluator_path}/paper/tester.py --checkpoint_path {model_dir} --eval segm")

    out_file_path = os.path.join(model_dir, output_file_name)

    total_out_file = open(out_file_path,"w+")
    print("Evaluating PDQ for: " + model_dir)
    total_out_file.write("Evaluating PDQ for: " + model_dir)
    total_out_file.close()

    step = 0.025
    for min_score in np.arange(0.0, 1.0 + step, step):
        total_out_file = open(out_file_path,"a+")
        print(f"\nScore threshold: {min_score:.3f}")
        total_out_file.write(f"\nScore threshold: {min_score:.3f}")
        total_out_file.close()
        eval_pdq_for_score_threshold(model_dir, min_score, out_file_path)
    total_out_file.close()

def prepare_pdq_dir(model_dir):
    pdq_eval_dir = os.path.join(model_dir,"pdq")
    if os.path.exists(pdq_eval_dir):
        shutil.rmtree(pdq_eval_dir) # remove all previous files
    os.makedirs(pdq_eval_dir)
    return pdq_eval_dir

def eval_pdq_for_score_threshold(model_dir, min_score_thrs, out_file_path):
    pdq_eval_dir = prepare_pdq_dir(model_dir)

    read_files.convert_coco_det_to_rvc_det(
        f"{model_dir}/{coco_dets_file_name}", 
        dataset_anns_path, 
        f"{pdq_eval_dir}/{rvc_dets_file_name}", 
        min_score_thrs)

    subprocess.call(["python", "evaluate.py", 
            "--test_set", "coco", 
            "--gt_loc", dataset_anns_path, 
            "--det_loc", f"{pdq_eval_dir}/{rvc_dets_file_name}", 
            "--save_folder", pdq_eval_dir,
            "--out_loc", out_file_path])

def make_graph(data):
    fig, ax = plt.subplots()
    markers = ["o","v","s","x"]
    lines = ["-",":","--"]
    colors = ["blue","green","gold","darkorange"]
    for i,[name,line] in enumerate(data):
        x,y = np.array(line).T
        ax.plot(x,y,marker = markers[i//3],markersize = 3,label = name,linestyle=lines[i%3],color=colors[i//3])
    ax.set_ylim(0,0.2)   
    ax.set_xlim(0,1)   
    ax.grid(visible=True)
    ax.set_xticks(np.arange(0,1.05,0.2))
    ax.set_yticks(np.arange(0,0.201,0.025))
    ax.set_xlabel("Confidence score threshold [-]")
    ax.set_ylabel("PDQ [-]")
    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.04,1),loc="upper left")
    plt.show()


def plot_data():
    data = []
    regex_pattern = f"threshold: (\d.\d+) - PDQ: (\d.\d+) - avg_pPDQ: (\d.\d+) - spatial_PDQ: (\d.\d+) - label_PDQ: (\d.\d+) - mAP: (\d.\d+) - TP: (\d+) - FP: (\d+) - FN: (\d+)"
    for dir in list_all_dirs_only(path_to_models):
        f = open(os.path.join(path_to_models, dir, output_file_name), "r")
        file_content = f.read()
        arch, channels = parse_config_and_channels_from_checkpoint_path(dir)
        arch_name = arch_translation[arch]
        channels_name = channels_translation[channels]
        model_data = []
        for match in re.finditer(regex_pattern, file_content):
            min_score = float(match.group(1))
            pdq = float(match.group(2))
            # avg_pPDQ = float(match.group(3))
            # spatial_PDQ = float(match.group(4))
            # label_PDQ = float(match.group(5))
            # map = float(match.group(6))
            # tp = float(match.group(7))
            # fp = float(match.group(8))
            # fn = float(match.group(9))
            single_score_thrs = [min_score, pdq]

            model_data.append(single_score_thrs)
        data.append([f"{arch_name} ({channels_name})", model_data])
    make_graph(data)

def evaluate_all():
    for dir in list_all_dirs_only(path_to_models):
        eval_pdq(os.path.join(path_to_models, dir))

if __name__ == "__main__":
    evaluate_all()
    # plot_data()