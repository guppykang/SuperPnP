from pathlib import Path
import subprocess
import numpy as np
import code



# from utils.eval_utils import Sc_Sfmleaner_frontend
class Sc_Sfmleaner_frontend(object):
    def __init__(self, BASE_DIR="./SC-SfMLearner-Release"):
        # BASE_DIR = "./SC-SfMLearner-Release"
        self.params = {}
        self.params["BASE_DIR"] = BASE_DIR
        self.dataset_dir = {}
        self.dataset_dir["euroc"] = f"./datasets/euroc/"
        self.dataset_dir["euroc_undistorted"] = f"./datasets/sequences_undistorted/"
        # self.dataset_dir["tum"] = f"./datasets/tum/raw_split_pseudoRgbd/"
        self.dataset_dir["tum"] = f"/jbk001-data1/datasets/tum"
        self.dataset_dir["kitti"] = f"/media/yoyee/Big_re/kitti/data_odometry_color/dataset/sequences/" # newly added
        pass

    def get_saved_trajectory(self, subfolder, model, dataset, sequence, trailing=""):
        if trailing == "":
            return f"./results/{subfolder}/{model}/{dataset}/{sequence}/{sequence}.txt"
        else:
            return f"./results/{subfolder}/{model}/{dataset}/{sequence}/{sequence}{trailing}"

    def get_saved_folder(self, subfolder, model, dataset, sequence):
        return str(Path('./results') / str(subfolder) / str(model) / str(dataset) / str(sequence))

    def get_saved_base(self, subfolder, model, dataset):
        return f"./results/{subfolder}/{model}/{dataset}/"

    # @staticmethod
    def get_command_scsfmlearner(self, args, save_folder, dataset, sequence="V1_01_easy",
                                 pretrained="/pretrained/pose/cs+k_pose.tar", 
                                 skip_frame=1, keyframe=""):
        BASE_DIR = self.params["BASE_DIR"] + "/"
        pyfile = BASE_DIR + "/test_vo.py"
        pretrained = BASE_DIR  + pretrained
        other_com = ""
        if keyframe != "":
            keyframe += f"{sequence}/{sequence}.txt_key"
            other_com += f" --keyframe {keyframe}"
        other_com += f" --lstm" if args.lstm else ""
        other_com += f" --config {args.config}" if args.config != "" else ""
        # save_folder = f"./results/{exp_name}/{dataset}/{sequence}"
            
        dataset_dir = self.dataset_dir[dataset]
        if dataset == "euroc" or dataset == "tum":
            ### euroc
            if args.undistorted:
                dataset_dir = self.dataset_dir["euroc_undistorted"]
                other_com += "--img-exts jpg"
            # sequence = sequence
            # dataset_dir = f"./datasets/euroc/"
            command = f"python {pyfile} -d {dataset} --img-height {args.height} --img-width {args.width} \
            --sequence {sequence} \
            --pretrained-posenet {pretrained} \
            --dataset-dir {dataset_dir} \
            --output-dir {save_folder}/ \
            --skip_frame {skip_frame} \
            {other_com}"
        elif dataset == "kitti":
            # dataset_dir = f"./datasets/kitti/sequences/"
            # dataset_dir = f"/media/yoyee/Big_re/kitti/sequences/"
            # dataset_dir = f"/media/yoyee/Big_re/kitti/data_odometry_color/dataset/sequences/" # newly added
            command = f"python {pyfile} -d {dataset} --img-height 256 --img-width 832 \
            --sequence {sequence} \
            --pretrained-posenet {pretrained} \
            --dataset-dir {dataset_dir} \
            --output-dir {save_folder}/ \
            --skip_frame {skip_frame}\
            {other_com}\
            --all_frame"

        return command


    # EuRoC dataset
    @staticmethod
    def get_times_file(BASE_DIR, seq):
        stmp = seq[0:2] + seq[3:5]
        path_to_times_file = (
            BASE_DIR
            + f"/orbslam2/Examples/Monocular/EuRoC_TimeStamps/{stmp}.txt"
        )
        return path_to_times_file

""" edited by youyi on Sep 2, 2020: for TrianFlow and others """
class flow_frontend(Sc_Sfmleaner_frontend):
    def __init__(self, model='superglueflow'):
        super().__init__('.')
        self.model = model
        self.pyFile = "infer_tum.py"
        pass
    def get_command_scsfmlearner(self, args, save_folder, dataset, sequence="V1_01_easy",
                             pretrained="/pretrained/pose/cs+k_pose.tar", 
                             skip_frame=1, keyframe=""):
        command = f"python {self.pyFile} --model {self.model} --sequence {sequence} \
                --traj_save_dir {save_folder} --iters {args.iters}"
        return command
        pass
    
    def get_saved_trajectory(self, subfolder, dataset, sequence, trailing=""):
        return f"./results/{subfolder}/{dataset}/{sequence}/{self.model}/preds.tum"
        # if trailing == "":
        #     return f"./results/{subfolder}/{model}/{dataset}/{sequence}/{sequence}.txt"
        # else:
        #     return f"./results/{subfolder}/{model}/{dataset}/{sequence}/{sequence}{trailing}"
    def get_saved_folder(self, subfolder, model, dataset, sequence, add_model=False):
        if add_model:
            return f"./results/{subfolder}/{model}/{dataset}/{sequence}/{self.model}"
        else:
            return f"./results/{subfolder}/{model}/{dataset}/{sequence}"
    
##########################
##### for evaluation #####
##########################
# from utils.eval_utils import Eval_frontend
class Eval_frontend(object):
    def __init__(self, plot_mode="xz", correct_scale=True, plot=False):
        self.params = {}
        self.params["plot_mode"] = plot_mode
        self.params["scale"] = "-s --align" if correct_scale else ""
        self.params["plot"] = "-p" if plot else ""
        pass

    def eval_trajectory(
        self, est_traj, gt_traj, mode="kitti", save_name="", save_folder=None, traj=True, ape=False, rpe=False
    ):
        command_list = []
        input_list = []
        params = self.params
        scale = params["scale"]
        plot_dis = params["plot"]
        plot_mode = params["plot_mode"]
        other_commands = "  -d 30 --all_pairs "
        unzip = True
        input_append = None # b"y"
        run_evo = True
        if save_folder is None:
            save_folder = Path(est_traj).parent
            save_print_f = save_folder / "results.txt"
            save_print_mode = "| tee -a"
        # ./datasets/euroc/V1_01_easy/mav0/data_f.kitti
        if traj:
            save_plot = save_folder / f"traj_{plot_mode}{save_name}.pdf"
            save_result = save_folder / f"traj_{plot_mode}{save_name}.zip"
            command = f"evo_traj {mode} {scale} {est_traj} --ref={gt_traj} {scale} {plot_dis} \
                        --plot_mode={plot_mode} --save_plot {save_plot}"
            if run_evo:
                command_list.append(command)
                input_list.append(input_append)
        if ape:
            save_plot = save_folder / f"ape_{plot_mode}{save_name}.pdf"
            save_result = save_folder / f"ape_{plot_mode}{save_name}.zip"
            command = f"evo_ape {mode} {scale} {gt_traj} {est_traj} {scale} {plot_dis} \
                        --plot_mode={plot_mode} --save_plot {save_plot} --save_results {save_result} \
                        {save_print_mode} {save_print_f}"
            if run_evo:
                command_list.append(command)
                input_list.append(input_append)
            if unzip:
                command_list.append(f"unar -d {save_result} -o {save_folder}")
                input_list.append(None)

        if rpe:
            save_plot = save_folder / f"rpe_{plot_mode}{save_name}.pdf"
            save_result = save_folder / f"rpe_{plot_mode}{save_name}.zip"
            # command = f"evo_rpe {mode} {scale} {gt_traj} {est_traj} {scale} {plot_dis} \
            #             --plot_mode={plot_mode} --save_plot {save_plot} --save_results {save_result} \
            #             {save_print_mode} {save_print_f} "
            command = f"evo_rpe {mode} {scale} {gt_traj} {est_traj} {scale}  \
                        --save_results {save_result} {other_commands} \
                        {save_print_mode} {save_print_f} "
            if run_evo:
                command_list.append(command)
                input_list.append(input_append)
            if unzip:
                command_list.append(f"unar -d {save_result} -o {save_folder}")
                input_list.append(None)

        return command_list, input_list
        #  evo_traj kitti -s results/vo/cs+k_pose/euroc/V1_01_easy.txt --ref=./datasets/euroc/V1_01_easy/mav0/data_f.kitti -s -p --plot_mode=xy --save_plot results/vo/cs+k_pose/euroc/traj_xy.zip
        pass


    @staticmethod
    def match_time_stamps(
        est_file, gt_file, save_folder=None, BASE_PATH="./tool", max_difference=0.1
    ):
        filter_ext = ".txt"
        if save_folder is None:
            save_folder = Path(est_file).parent
            # save_print_f = save_folder/"results.txt"
        m_est_file, m_gt_file = (
            save_folder / f"est_m_kitti{filter_ext}",
            save_folder / f"gt_m_kitti{filter_ext}",
        )
        # match poses
        print("match est to gt")
        subprocess.run(
            f"python {BASE_PATH}/associate.py {gt_file} {est_file} --max_difference {max_difference} \
                --save_file {m_est_file}",
            shell=True,
            check=True,
        )
        # match poses
        print("match gt to est")
        subprocess.run(
            f"python {BASE_PATH}/associate.py {est_file} {gt_file} --first_only --max_difference 0.10 \
                --save_file {m_gt_file}",
            shell=True,
            check=True,
        )

        # est_poses = np.genfromtxt(m_est_file, delimiter=" ")[:,1:]
        ## convert gt_file to csv format
        # filename = f"{folder}/{gt_file[:-4]+filter_ext}"
        # file = np.loadtxt(filename)
        # np.savetxt(filename, file, fmt="%s", delimiter=",")
        return {"est_file": m_est_file, "gt_file": m_gt_file}

    @staticmethod
    def kitti_wTime_tum(t_pose_file, invert=False, reset_stamp=False):
        # from KITTI_odometry_evaluation_tool.tools.pose_evaluation_utils import rot2quat
        from tool.pose_evaluation_utils import rot2quat
        def matrix_to_quatVec(mat):
            # print(f"mat: {mat}")
            rotation = mat[:3,:3]
            trans = mat[:3,3]
            # from pyquaternion import Quaternion
            # qua = Quaternion(matrix=rotation)
            # vect = np.concatenate((trans, qua.elements), axis=0)
            qua = rot2quat(rotation)
            vect = np.concatenate((trans, qua), axis=0)
            return vect
        def mat_invert(mat):
            if mat.shape[0] != 3:
                raise "wrong shape"
            else:
                row = np.array([[0,0,0,1]])
                mat = np.concatenate((mat, row), axis=0)
            from numpy.linalg import inv
            mat_inv = inv(mat)
            return mat_inv[:3]
                            

        # save parameters
        save_folder = Path(t_pose_file).parent
        filename = Path(t_pose_file).stem + ".tum"
        # load and process
        t_pose = np.genfromtxt(t_pose_file, delimiter=" ")
        if reset_stamp:
            stamps = np.arange(t_pose.shape[0]).reshape((-1,1))
        else:
            stamps = t_pose[:,:1]
        poses = t_pose[:,1:]
        if invert:
            pose_qua = np.array([matrix_to_quatVec(mat_invert(m.reshape(3,4))) for m in poses])
        else:
            pose_qua = np.array([matrix_to_quatVec(m.reshape(3,4)) for m in poses])
        pose_tum = np.concatenate((stamps, pose_qua), axis=1)
        # save file
        np.savetxt(save_folder/filename, pose_tum, fmt="%s", delimiter=" ")
        return save_folder/filename

    

# from utils.eval_utils import Euroc_dataset
class Euroc_dataset(object):
    def __init__(self):
        self.train_seqs = [
            "MH_01_easy",
            "MH_02_easy",
            "MH_04_difficult",
            "V1_01_easy",
            "V1_02_medium",
            "V1_03_difficult",
        ]
        self.test_seqs = [
            "MH_05_difficult",
            "V2_01_easy",
            "V2_02_medium",
            "V2_03_difficult",
            "MH_03_medium",
        ]
        pass

    def path_params(self):
        pass

    def get_all_seqs(self):
        return self.train_seqs + self.test_seqs

    def process_gt_poses(self, dataset="./datasets/euroc"):
        command = f"python tool/process_poses_euroc.py --dataset_dir {dataset} --dataset euroc"
        return command
        # print(f"command: {command}")

    def extract_zips(self, seqs):
        import subprocess

        for s in seqs:
            command = f"unar -d {s}.zip"
            subprocess.run(f"{command}", shell=True, check=True)

    def get_seq_gt_filename(self, seq, processed=True, dataset="./datasets/euroc", with_time=False):
        if processed:
            ext = "_wTime" if with_time else ""
            file = f"{dataset}/{seq}/mav0/data_f{ext}.kitti"
        else:
            NotImplementedError
        return file


class Tum_dataset(Euroc_dataset):
    def __init__(self):
        ## use scsfm split
        self.train_seqs = [ # only process train_set
        'rgbd_dataset_freiburg2_desk',
        'rgbd_dataset_freiburg2_360_kidnap',
        'rgbd_dataset_freiburg2_pioneer_360',
#         'rgbd_dataset_freiburg2_pioneer_slam3',
        'rgbd_dataset_freiburg3_large_cabinet',
        'rgbd_dataset_freiburg3_sitting_static', 
        'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        'rgbd_dataset_freiburg3_structure_notexture_far', 
        'rgbd_dataset_freiburg3_structure_texture_far'
        
        ]
        self.test_seqs = [
        ]        
        # self.train_seqs = [ # only process train_set
        #     "rgbd_dataset_freiburg3_long_office_household",
        #     "rgbd_dataset_freiburg3_long_office_household_validation",
        #     "rgbd_dataset_freiburg3_sitting_xyz",
        #     "rgbd_dataset_freiburg3_structure_texture_far",
        #     "rgbd_dataset_freiburg3_structure_texture_near",
        #     "rgbd_dataset_freiburg3_teddy",
        # ]
        # self.test_seqs = [
        #     "rgbd_dataset_freiburg3_walking_xyz",
        #     "rgbd_dataset_freiburg3_large_cabinet_validation",
        # ]
        
        # super.__init__()
        # self.train_seqs = [
        #     "rgbd_dataset_freiburg1_desk",
        #     "rgbd_dataset_freiburg1_room",
        #     "rgbd_dataset_freiburg2_desk",
        #     "rgbd_dataset_freiburg3_long_office_household",
        # ]
        # self.test_seqs = [
        #     "rgbd_dataset_freiburg1_desk2",
        #     "rgbd_dataset_freiburg2_xyz",
        #     "rgbd_dataset_freiburg3_nostructure_texture_far",
        # ]
        pass
    def get_seq_gt_filename(self, seq, processed=False, dataset="./datasets/tum", with_time=False):
        if processed:
            NotImplementedError
        else:
            file = f"{dataset}/{seq}/groundtruth.txt"
        return file

# from utils.eval_utils import Orb_slam_frontend
class twitchslam_frontend(Sc_Sfmleaner_frontend):
    def __init__(self, BASE_DIR="./"):
        super().__init__(self)
        pass

    def get_g2oOptim_commands(self, base_pose, ref_pose, output_path, seq, align_mode='rel'):
        # python pose_graph_g2o_SE3.py \
        # --base_pose results/scsfm/scsfm_posenet_256_kf_rot_thd0.5_200k_eval_rot0.5_allKey/kitti/10/10.txt \
        # --ref_pose datasets/gt_poses/10.txt \
        # --output_pose results/scsfm_opt/scsfm-kitti_gt-opt/kitti/10/ --seq 10 \
        # --align_mode abs
        ref_pose = " ".join(ref_pose)
        command = f"python pose_graph_g2o_SE3.py --base_pose {base_pose} \
                --ref_pose {ref_pose} --output_pose {output_path} --seq {seq} \
                --align_mode {align_mode}"
        return command
        pass

# from utils.eval_utils import Orb_slam_frontend
class Orb_slam_frontend(Sc_Sfmleaner_frontend):
    def __init__(self, BASE_DIR="./"):
        pass
    



# from utils.eval_utils import Result_processor
class Result_processor(object):
    def __init__(self, result_dict_entry):
        self.result_dict_entry = result_dict_entry
        self.result_arr = None
        pass

    def get_result_arr(self, result_entries, result_tool="", entry='rmse'):
        results = []
        print(f"result_tool: {result_tool}")
        for i, en in enumerate(result_entries):
            print(f"{en}: {self.result_dict_entry[en]}")
            if result_tool == 'evo':
                results.append(self.result_dict_entry[en][entry])
            elif result_tool == 'snippet':
                results.append(self.result_dict_entry[en][entry])
            else:
                results.append(self.result_dict_entry[en][:, 1:].flatten())
        results = np.array(results)  # .reshape(-1, len(result_entries))
        results = results[..., np.newaxis] if results.ndim==1 else results
        return results

    def get_latex_from_arr(self, result_arr):
        table_body = []
        for i, line in enumerate(result_arr):
            li = self.list_to_style("", line)
            table_body.append(li)
        return table_body

    def get_average(self, result_arr, col=True):
        ax = 1 if col else 0
        ave = result_arr.mean(axis=ax)
        result_arr = np.concatenate((result_arr, ave.reshape(-1, 1)), axis=ax)
        return result_arr

    @staticmethod
    def list_to_style(item, a_list, seperator=" & "):
        line = f"{item} & " + seperator.join([f"{i:.3f}" for i in a_list]) + "\n"
        line += "\\\\ \\hline"
        return line

    @staticmethod
    def result_reader(model_fe, subfolder, model, dataset, seqs, metric = "rpe_xy", filename="stats.json"): #  "ape_xy"
        result_entries = []
        result_table = {}
        def load_json(file_name):
            import json
            with open(file_name) as json_file:
                data = json.load(json_file)
            return data
        for i, seq in enumerate(seqs):
            save_folder = model_fe.get_saved_folder(subfolder, model, dataset, seq)
            result_file = f"{save_folder}/{metric}/{filename}"
            if Path(result_file).exists():
                result_entries.append(seq)
                result_table[seq] = load_json(result_file)
            else:
                print(f"file: {result_file} doesn't exist")
        return {'result_entries': result_entries, 'result_table': result_table}


    def add_result_table(self, dataset, subfolder, model,  # result_folder
            sequences, model_fe, metric="ape_xy", snippet=False):
        result_entries = []  # for args.table
        result_table = {}
        if dataset == 'kitti' and snippet == False:
            for seq in sequences:
                save_folder = model_fe.get_saved_folder(subfolder, model, dataset, seq) # get_saved_folder(result_folder, dataset, seq)
                result_file = f"{save_folder}/result.txt"
                print(f"result_file: {result_file}")
                if Path(result_file).exists():
                    result_entries.append(seq)
                    print(f"result_file: {result_file}, result_entries: {result_entries}")
                    result_table[seq] = np.genfromtxt(result_file, delimiter=":")
        elif (dataset == 'kitti' or dataset == 'euroc') and snippet == True: # euroc is fine as well
            data = self.result_reader(model_fe, subfolder, model, dataset, sequences, metric='.', filename='snip_ate.yml')
            result_entries = data['result_entries']
            result_table = data['result_table']            
        else:
            # metric = "ape_xy" # "ape_xy"
            data = self.result_reader(model_fe, subfolder, model, dataset, sequences, metric)
            result_entries = data['result_entries']
            result_table = data['result_table']
        self.result_dict_entry = result_table
        return {'result_entries': result_entries, 'result_table': result_table}
