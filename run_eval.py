## euler angle to 3x4 matrix
## or quaternion to 3x4 matrix
## preprocess before kitti evaluation.

## orbslam2 python module
import sys
import os

# from tools.pose_evaluation_utils import quat_pose_to_mat
import argparse
import yaml
import numpy as np
import subprocess
from pathlib import Path
from glob import glob
import logging
import code


from utils.eval_utils import flow_frontend
from utils.eval_utils import Sc_Sfmleaner_frontend
from utils.eval_utils import Eval_frontend
from utils.eval_utils import Result_processor
from utils.eval_utils import Euroc_dataset, Tum_dataset
# from utils.eval_utils import Orb_slam_frontend, twitchslam_frontend



def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def euler2mat(vec):
    # print(f"vec: {vec}")
    rot = eulerAnglesToRotationMatrix(vec[:3])
    # print(f"rot: {rot}, trans: {vec[3:]}")
    mat = np.concatenate((rot, vec[3:].reshape(3, 1)), axis=1)
    return mat


def eval_trajectory(save_folder, alignment="7dof"):
    command = f"python kitti-odom-eval/eval_odom.py --result {save_folder} --align {alignment}"
    print(f"run ==> {command}")
    subprocess.run(f"{command}", shell=True, check=True)
    pass

def eval_trajectory_snippet(save_folder, seq, length=5):
    from deepsfm_dummy.utils.eval_tools import Exp_table_processor
    # seq = "10"
    table_processor = Exp_table_processor
    poses_gt = table_processor.read_gt_poses(path='./datasets/kitti/poses/', seq=seq)
    poses_est = np.genfromtxt(f'{save_folder}/{seq}.txt')
    poses_est = poses_est[:,1:].reshape(-1,12)
    poses_est = poses_est.reshape(-1,3,4)
    print(f"length est vs. gt: {len(poses_est)}, {len(poses_gt)}")
    assert len(poses_est) == len(poses_gt)
    data = table_processor.pose_seq_ate(poses_est, poses_gt, 5)
    entries = ["error_names", "mean_errors", "std_errors"]
    # results = { key: data[key] for key in entries }
    results = {}
    for i, n in enumerate(data["error_names"]):
        for item in entries[1:]:
            results[f"{n}_{item}"] = data[item][i].astype(float)
    dump_json(results, f"{save_folder}/snip_ate.yml")
    # print(data)    
    pass

def eval_trajectory_snippet_seqs(poses_est, poses_gt, save_folder, length=5):
    from deepsfm_dummy.utils.eval_tools import Exp_table_processor
    table_processor = Exp_table_processor
    # seq = "10"
    # poses_gt = table_processor.read_gt_poses(path='./datasets/kitti/poses/', seq=seq)
    # poses_est = np.genfromtxt(f'{save_folder}/{seq}.txt')
    # poses_est = poses_est[:,1:].reshape(-1,12)
    # poses_est = poses_est.reshape(-1,3,4)
    print(f"length est vs. gt: {len(poses_est)}, {len(poses_gt)}")
    assert len(poses_est) == len(poses_gt)
    data = table_processor.pose_seq_ate(poses_est, poses_gt, 5)
    entries = ["error_names", "mean_errors", "std_errors"]
    # results = { key: data[key] for key in entries }
    results = {}
    for i, n in enumerate(data["error_names"]):
        for item in entries[1:]:
            results[f"{n}_{item}"] = data[item][i].astype(float)
    dump_json(results, f"{save_folder}/snip_ate.yml")
    # print(data)    
    pass

def get_sequences(args):
    sequences = []
    controller = None
    if args.dataset == "kitti":
        if args.seq == "all":
            sequences = [f"{seq:02}" for seq in range(11)]
        else:
            sequences = [args.seq]
    elif args.dataset == 'euroc':
        euroc_controller = Euroc_dataset()
        if args.seq == "all":
            ## manually run the command if needed
            command = euroc_controller.process_gt_poses()
            print(f"process datasets: {command}")
            print(f"+++++  manually run the command if needed  +++++")
            sequences = euroc_controller.get_all_seqs()
        else:
            sequences = [args.seq]
        controller = euroc_controller
    elif args.dataset == 'tum':
        controller = Tum_dataset()
        if args.seq == "all":
            sequences = controller.get_all_seqs()
        else:
            sequences = [args.seq]
    else:
        logging.error(f"dataset: {dataset} is not defined.")
        pass
    return sequences, controller

def dump_json(dict, filename):
    import json
    json = json.dumps(dict)
    f = open(filename, "w")
    f.write(json)
    f.close()

def dump_config(config, output_dir, filename='config.yml'):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    pass

# for orbslam kitti config
def get_config_file(seq, dataset):
    config_file = ""
    if dataset == "kitti":
        seq = int(seq)
        if seq <= 2:
            config_file = "KITTI00-02.yaml"
        elif seq == 3:
            config_file = "KITTI03.yaml"
        else:
            config_file = "KITTI04-12.yaml"
    elif dataset == "euroc":
        config_file = "EuRoC.yaml"
    return config_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change format from quaternion to kitti format"
    )
    # subparsers = parser.add_subparsers(help='sub-command help')
    # parser_optim = parser.add_argument_group('optim', 'use for g2o optimization')

    parser.add_argument(
        "exper_name", type=str, help="The experiment name for exporting results"
    )
    parser.add_argument(
        "-sub", "--subfolder", type=str, default="", help="result subfolder, can be separated using model name"
    )
    models = ['superglueflow', 'siftflow', 'siftflow_scsfm', 'superglueflow', 'superflow', 'superflow2']
    print(f"models: {models} are supported")
    
    parser.add_argument(
        "-m", "--model", type=str, default="siftflow", choices=models, help="model to test"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="kitti", help="[kitti |  euroc | tum ... ]"
    )
    parser.add_argument('--iters', type=int, default='-1', help='Limited iterations for debugging')
    
    parser.add_argument(
        "--undistorted", action="store_true", default=False, help="Must be Euroc dataset"
    )
    
    # parser.add_argument("--dataset_dir", type=str, default="", help='link to dataset')
    # parser.add_argument('out_file',     type=str,  help='the output file name')
    # parser.add_argument('--result_dir', type=str, default='./data/',              help='Directory path of storing the odometry results')
    # parser.add_argument('--action',  type=str, default=None, help='[ euler2mat | ]')
    # parser.add_argument('--toCameraCoord',   type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to convert the pose to camera coordinate')
    parser.add_argument("--eval", action="store_true", help="eval the sequences")
    parser.add_argument("--snippet", action="store_true", help="eval the sequences with snippets")
    parser.add_argument(
        "--run", action="store_true", help="run to evaluate the sequences"
    )
    # g2o optimization
    # parser_optim.add_argument(
    #     "--optim", action="store_true", help="optimize the pose using ref pose in g2o"
    # )
    # parser_optim.add_argument(
    #     "--config", dest="config", default="", 
    #     help="Can specify all the setting in the file"
    # )
    # parser_optim.add_argument(
    #     "--base_pose", type=str, default="", help="base pose (input base path .../seq/seq.txt) in g2o"
    # )
    # parser_optim.add_argument(
    #     "--ref_poses", nargs='+', default="", help="list of ref pose (input base path .../seq/seq.txt) in g2o"
    # )
    # --output_path ...
    

     
    parser.add_argument(
        "--table", action="store_true", help="collect the evaluation to table"
    )
    parser.add_argument(
        "--view", action="store_true", default=False, help="view the slam from the display"
    )
    parser.add_argument(
        "--wTime", action="store_true", default=False, help="append time stamps to the results"
    )

    parser.add_argument(
        "--seq", type=str, default="all", help="run on all sequences"
    )
    ## for scsfm 
    parser.add_argument("--skip_frame", default=1, type=int, help="The time differences between frames")
    # pretrained model
    parser.add_argument(
        "--pretrained", type=str, default="./pretrained/pose/cs+k_pose.tar", help="path of trained model"
    )
    parser.add_argument("--keyframe", default="", type=str, help="File with keyframe stamps")
    # eval
    parser.add_argument(
        "--metric", type=str, default="ape_xy", help="EuRoc: [ape_xy | rpe_xy], Kitti/euroc snippet: [mean | std]"
    )

    # lstm network
    parser.add_argument("--lstm", action='store_true', default=False, help="use lstm network")
    
    BASE_DIR = "/home/yoyee/Documents/deep_keyframe"
    # BASE_DIR = "/home/yyjau/Documents/deep_keyframe"
    # BASE_DIR = "."
    args = parser.parse_args()
    print(f"args: {args}")
    ## parameters
    result_entries = []  # for args.table
    result_table = {}
    re_processor = Result_processor(None)

    dataset = args.dataset
    subfolder = args.subfolder
    if args.undistorted:
        assert args.dataset == "euroc"

    # dataset controller
    # if dataset == 'euroc':
    #     euroc_controller = Euroc_dataset()
    seqs, controller = get_sequences(args)

    w_time = args.wTime
    print(f"w_time: {w_time}")
    args.exper_name = args.exper_name + "_t" if args.wTime else args.exper_name

    if args.model in models:
        model_fe = flow_frontend(args.model)
    
    if args.run:
        ## kitti and euroc are the same
        # seqs = sequences
        if dataset == 'euroc':
            args.width, args.height = 752, 480
        elif dataset == 'tum':
            args.width, args.height = 640, 480

        dump_config(args, model_fe.get_saved_base(subfolder, args.exper_name, dataset))
        for s in seqs:
#             save_folder = model_fe.get_saved_folder(subfolder, args.exper_name, dataset, s)
            save_folder = model_fe.get_saved_folder(subfolder, args.exper_name, dataset, '')
            command = model_fe.get_command_scsfmlearner(
                args,
                save_folder, dataset, sequence=s, skip_frame=args.skip_frame,
                pretrained=args.pretrained,
                keyframe=args.keyframe
            )
            print(f"command: {command}")
            subprocess.run(f"{command}", shell=True, check=True)
    
    if args.eval:
        # eval_fe = Eval_frontend(plot_mode="xy", plot=False)
        eval_fe = Eval_frontend(plot_mode="xy", plot=False)
        for i, s in enumerate(seqs):
            save_folder = model_fe.get_saved_folder(subfolder, args.exper_name, dataset, s)
            if dataset == 'kitti':
                alignment = '7dof'
                eval_trajectory(save_folder, alignment=alignment)                        
                eval_trajectory_snippet(save_folder, s, length=5)
            elif dataset == 'euroc':
                est_traj = model_fe.get_saved_trajectory(subfolder, args.exper_name, dataset, s, trailing="_noTime.txt")
                gt_traj = controller.get_seq_gt_filename(s)
                # test snippet
                print(f"est_traj: {est_traj}, gt_traj: {gt_traj}, save_folder: {save_folder}")
                poses_est = np.genfromtxt(est_traj)
                poses_est = poses_est.reshape(-1,12)
                poses_est = poses_est.reshape(-1,3,4)
                poses_gt = np.genfromtxt(gt_traj).reshape(-1,12).reshape(-1,3,4)
                eval_trajectory_snippet_seqs(poses_est, poses_gt, save_folder, length=5)
                if_evo = False
                if if_evo:
                    # use evo
                    command_list, input_list = eval_fe.eval_trajectory(
                        est_traj, gt_traj, traj=True, ape=True, rpe=True
                    )
                    print(f"eval: {command_list}")
                    for command, inp in zip(command_list, input_list):
                        subprocess.run(f"{command}", shell=True, check=True, input=inp)
                # break
            else:
                # convert to tum
                seq = s
                est_tum = model_fe.get_saved_trajectory(subfolder, args.exper_name, 
                            dataset, seq, trailing='')
                # get ground truth file
                gt_traj = controller.get_seq_gt_filename(seq, dataset=model_fe.dataset_dir[dataset], with_time=True)

                # est_tum = eval_fe.kitti_wTime_tum(est_traj)
                if dataset == 'euroc':
                    gt_tum = eval_fe.kitti_wTime_tum(gt_traj, reset_stamp=True)
                else:
                    gt_tum = gt_traj
                print(f"est_tum: {est_tum}, gt_tum: {gt_tum}")
                # print(f"est_traj: {est_traj}, gt_traj: {gt_traj}, save_folder: {save_folder}")
                # poses_est = np.genfromtxt(est_traj)
                # poses_est = poses_est.reshape(-1,12)
                # poses_est = poses_est.reshape(-1,3,4)
                # poses_gt = np.genfromtxt(gt_traj).reshape(-1,12).reshape(-1,3,4)
                # eval_trajectory_snippet_seqs(poses_est, poses_gt, save_folder, length=5)
                if_evo = True
                if if_evo:
                    ## get commands to run
                    eval_mode = "tum"
                    command_list, input_list = eval_fe.eval_trajectory(
                        est_tum, gt_tum, mode=eval_mode, traj=True, ape=True, rpe=True
                    )
                    print(f"eval: {command_list}")
                    for command, inp in zip(command_list, input_list):
                        subprocess.run(f"{command}", shell=True, check=True, input=inp)
    
    
    
    ## deprecated (you-yi)
    if args.model == "scsfm":
        model_fe = Sc_Sfmleaner_frontend()
        # seqs = sequences
        # pretrained = "./pretrained/pose/cs+k_pose.tar"
        if args.run:
            ## kitti and euroc are the same
            if dataset == 'euroc':
                args.width, args.height = 752, 480
            elif dataset == 'tum':
                args.width, args.height = 640, 480
                 
            dump_config(args, model_fe.get_saved_base(subfolder, args.exper_name, dataset))
            for s in seqs:
                save_folder = model_fe.get_saved_folder(subfolder, args.exper_name, dataset, s)
                command = model_fe.get_command_scsfmlearner(
                    args,
                    save_folder, dataset, sequence=s, skip_frame=args.skip_frame,
                    pretrained=args.pretrained,
                    keyframe=args.keyframe
                )
                print(f"command: {command}")
                subprocess.run(f"{command}", shell=True, check=True)
          
        if args.eval:
            # eval_fe = Eval_frontend(plot_mode="xy", plot=False)
            eval_fe = Eval_frontend(plot_mode="xy", plot=False)
            for i, s in enumerate(seqs):
                save_folder = model_fe.get_saved_folder(subfolder, args.exper_name, dataset, s)
                if dataset == 'kitti':
                    alignment = '7dof'
                    eval_trajectory(save_folder, alignment=alignment)                        
                    eval_trajectory_snippet(save_folder, s, length=5)
                elif dataset == 'euroc':
                    est_traj = model_fe.get_saved_trajectory(subfolder, args.exper_name, dataset, s, trailing="_noTime.txt")
                    gt_traj = controller.get_seq_gt_filename(s)
                    # test snippet
                    print(f"est_traj: {est_traj}, gt_traj: {gt_traj}, save_folder: {save_folder}")
                    poses_est = np.genfromtxt(est_traj)
                    poses_est = poses_est.reshape(-1,12)
                    poses_est = poses_est.reshape(-1,3,4)
                    poses_gt = np.genfromtxt(gt_traj).reshape(-1,12).reshape(-1,3,4)
                    eval_trajectory_snippet_seqs(poses_est, poses_gt, save_folder, length=5)
                    if_evo = False
                    if if_evo:
                        # use evo
                        command_list, input_list = eval_fe.eval_trajectory(
                            est_traj, gt_traj, traj=True, ape=True, rpe=True
                        )
                        print(f"eval: {command_list}")
                        for command, inp in zip(command_list, input_list):
                            subprocess.run(f"{command}", shell=True, check=True, input=inp)
                    # break
                else:
                    # convert to tum
                    seq = s
                    est_traj = model_fe.get_saved_trajectory(subfolder, args.exper_name, 
                                dataset, seq, trailing='_t.txt')
                    # get ground truth file
                    gt_traj = controller.get_seq_gt_filename(seq, dataset=model_fe.dataset_dir[dataset], with_time=True)

                    est_tum = eval_fe.kitti_wTime_tum(est_traj)
                    if dataset == 'euroc':
                        gt_tum = eval_fe.kitti_wTime_tum(gt_traj, reset_stamp=True)
                    else:
                        gt_tum = gt_traj
                    print(f"est_tum: {est_tum}, gt_tum: {gt_tum}")
                    # print(f"est_traj: {est_traj}, gt_traj: {gt_traj}, save_folder: {save_folder}")
                    # poses_est = np.genfromtxt(est_traj)
                    # poses_est = poses_est.reshape(-1,12)
                    # poses_est = poses_est.reshape(-1,3,4)
                    # poses_gt = np.genfromtxt(gt_traj).reshape(-1,12).reshape(-1,3,4)
                    # eval_trajectory_snippet_seqs(poses_est, poses_gt, save_folder, length=5)
                    if_evo = True
                    if if_evo:
                        ## get commands to run
                        eval_mode = "tum"
                        command_list, input_list = eval_fe.eval_trajectory(
                            est_tum, gt_tum, mode=eval_mode, traj=True, ape=True, rpe=True
                        )
                        print(f"eval: {command_list}")
                        for command, inp in zip(command_list, input_list):
                            subprocess.run(f"{command}", shell=True, check=True, input=inp)
                            

    if args.table:
        # metric 
        # result_arr = re_processor.get_result_arr(result_entries, result_tool="evo")
        def get_result_tools(dataset, args):
            result_tool = "" if dataset == 'kitti' else 'evo'
            if dataset == 'euroc' and args.snippet == True:
                result_tool = 'snippet'
            if dataset == 'kitti' and args.snippet == True:
                result_tool = 'snippet'

            ## result entry
            result_entry = "" 
            if result_tool == "evo":
                result_entry = "rmse"
            elif result_tool == "snippet":
                if args.metric == "mean" or args.metric == "std":
                    result_entry = f"ATE_{args.metric}_errors"
                else:
                    result_entry = "ATE_mean_errors"
                    print(f"metric *{args.metric}* is not valid. Use *{result_entry}*")
            return {'result_tool': result_tool, 'entry': result_entry}

        # get data
        data = re_processor.add_result_table(dataset, 
            subfolder, args.exper_name, seqs, model_fe, metric=args.metric, snippet=args.snippet)
        result_entries = data['result_entries']
        # print(f"===== result_tool: {result_tool} =====")
        # print(f"===== result_entries: {result_entries} =====")
        # result_arr = re_processor.get_result_arr(result_entries, result_tool=result_tool, entry=result_entry)
        result_tools_dict = get_result_tools(dataset, args)
        print(f"===== result_tools_dict: {result_tools_dict} =====")
        result_arr = re_processor.get_result_arr(result_entries, **result_tools_dict)
        result_arr = result_arr.transpose()
        result_arr = re_processor.get_average(result_arr)
        print(f"{result_table}, arr: {result_arr}")
        table_body = re_processor.get_latex_from_arr(result_arr)
        table_ready = "\n".join(table_body)
        print(f"=====metric: {args.metric}=====")
        print(f"titles: {' & '.join(result_entries)}")
        print(f"table_body: {table_ready}")

    # if args.action == 'euler2mat':
    #     poses_mat = [euler2mat(v)[:3,:] for v in poses_quat]
    #     poses_mat = np.array(poses_mat)
    #     poses_mat = poses_mat.reshape(-1, 12)
    # np.savetxt(args.out_file, poses_mat)

    # pose_eval = kittiOdomEval(args)
    # pose_eval.eval(toCameraCoord=args.toCameraCoord)   # set the value according to the predicted results
