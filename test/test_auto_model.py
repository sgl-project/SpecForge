import unittest
import shutil
from sgl_eagle.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
from sgl_eagle.modeling.auto_model import AutoModelForCausalLM

class TestAutoModelForCausalLM(unittest.TestCase):
    
    def test_automodel(self):
        """测试模型初始化"""
        model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-4-Maverick-17B-128E-Eagle3")
        self.assertIsInstance(model, LlamaForCausalLMEagle3)

if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTest(unittest.makeSuite(TestAutoModelForCausalLM))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
