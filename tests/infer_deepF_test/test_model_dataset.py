import subprocess
import pytest
# class TestClass:
#     def test_one(self):
#         x = "this"
#         assert "h" in x

#     def test_two(self):
#         x = "hello"
#         assert hasattr(x, "check")

# values_list=['apple','tomatoes','potatoes']
ModelList=["siftflow", "siftflow_deepF"]
class TestClass:

    @pytest.mark.parametrize("model", ModelList)
    def test_xxx(self,model):
        DatasetList = ["kitti", "tum"]
        #assert value==value
        for dataset in DatasetList:
            print(f"{model}/{dataset}")
            command = f"/jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_model_dataset -m {model} -d {dataset} --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10 --debug"
            print(f"run command: {command}")
            subprocess.run(f"{command}", shell=True, check=True)

    def pytest_collection_modifyitems(items):
        for item in items:
            # check that we are altering a test named `test_xxx`
            # and it accepts the `value` arg
            if item.originalname == 'test_xxx' and 'model' in item.fixturenames:
                item._nodeid = item.nodeid.replace(']', '').replace('xxx[', '')

#############################################                
############### deprecated below ############                
#############################################
                
#     def test_model_dataset(self):
#         ModelList=["siftflow", "siftflow_deepF"]
#         DatasetList = ["kitti", "tum"]
#         for model in ModelList:
#             for dataset in DatasetList:
#                 print(f"{model}/{dataset}")
#                 command = f"/jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_model_dataset -m {model} -d {dataset} --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10 --debug"
#                 print(f"run command: {command}")
# #                 subprocess.run(f"{command}", shell=True, check=True)
        
        
# def run_script(model, dataset):
#     print(f"test: {model}/{dataset}")
# #     command = f"/jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py test_model_dataset -m {model} -d {dataset} --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10 --debug"
# #     print(f"run command: {command}")
# #     subprocess.run(f"{command}", shell=True, check=True)
        
# if __name__ == '__main__':
#     ModelList=["siftflow", "siftflow_deepF"]
#     DatasetList = ["kitti", "tum"]
#     for model in ModelList:
#         for dataset in DatasetList:            
#             the_name = f"{model}_{dataset}" 
#             setattr(TestClass, the_name, classmethod(run_script(model, dataset)))
        
# class TestClass:
#     def test_model_dataset():
#         ModelList=["superflow", "siftflow", "siftflow_deepF", "siftflow_scsfm", "superglueflow_scsfm", "superglueflow", "trianflow"]
#         ModelList=["superflow", "siftflow"]
#         DatasetList=["kitti", "tum"]

#         for model in ModelList:
#             for dataset in DatasetList:
#                 print(f"{model}/ {dataset}"
#                 command = f"/jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/python run_eval.py -m {model} -d {dataset} --run -py /jbk001-data1/yyjau/conda/py36-superpnp_deepF/bin/ --iter 10 --debug"
