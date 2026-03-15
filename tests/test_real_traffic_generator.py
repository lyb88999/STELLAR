import unittest
import os
import torch
import numpy as np
import pandas as pd
import tempfile
from data_simulator.real_traffic_generator import RealTrafficGenerator, TrafficFlowDataset

class TestRealTrafficGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """创建测试用的CSV文件"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.csv_files = []

        n_samples = 20

        normal_data = {
            'Duration': np.linspace(0.1, 1.0, n_samples),
            'Size': np.linspace(100, 1000, n_samples),
            'Protocol': np.random.choice([6, 17], n_samples),
            'Sinr': np.linspace(10, 20, n_samples),
            'Throughput': np.linspace(5, 10, n_samples),
            'Flow_bytes_s': np.linspace(100, 1000, n_samples),
            'Flow_packets_s': np.linspace(10, 100, n_samples),
            'Inv_mean': np.linspace(0.01, 0.1, n_samples),
            'Inv_min': np.linspace(0.001, 0.01, n_samples),
            'Inv_max': np.linspace(0.1, 1.0, n_samples),
            'DNS_query_id': np.arange(1, n_samples+1),
            'L7_protocol': np.full(n_samples, 7),
            'DNS_type': np.full(n_samples, 1),
            'TTL_min': np.full(n_samples, 64),
            'TTL_max': np.full(n_samples, 64),
            'DNS_TTL_answer': np.full(n_samples, 3600),
            'Next_Current_diff': np.full(n_samples, 0.1),
            'Next_Pre_diff': np.full(n_samples, 0.2),
            'SNext_Current_diff': np.full(n_samples, 0.3),
            'SNext_Pre_diff': np.full(n_samples, 0.4),
            'Label': ['normal'] * n_samples
        }

        malicious_data = {
            'Duration': np.linspace(1.1, 2.0, n_samples),
            'Size': np.linspace(1100, 2000, n_samples),
            'Protocol': np.random.choice([6, 17], n_samples),
            'Sinr': np.linspace(21, 30, n_samples),
            'Throughput': np.linspace(11, 20, n_samples),
            'Flow_bytes_s': np.linspace(1100, 2000, n_samples),
            'Flow_packets_s': np.linspace(110, 200, n_samples),
            'Inv_mean': np.linspace(0.11, 0.2, n_samples),
            'Inv_min': np.linspace(0.011, 0.02, n_samples),
            'Inv_max': np.linspace(1.1, 2.0, n_samples),
            'DNS_query_id': np.arange(n_samples+1, 2*n_samples+1),
            'L7_protocol': np.full(n_samples, 7),
            'DNS_type': np.full(n_samples, 1),
            'TTL_min': np.full(n_samples, 64),
            'TTL_max': np.full(n_samples, 64),
            'DNS_TTL_answer': np.full(n_samples, 3600),
            'Next_Current_diff': np.full(n_samples, 0.6),
            'Next_Pre_diff': np.full(n_samples, 0.7),
            'SNext_Current_diff': np.full(n_samples, 0.8),
            'SNext_Pre_diff': np.full(n_samples, 0.9),
            'Label': ['malicious'] * n_samples
        }

        normal_df = pd.DataFrame(normal_data)
        malicious_df = pd.DataFrame(malicious_data)

        normal_file = os.path.join(cls.temp_dir.name, "normal_traffic.csv")
        malicious_file = os.path.join(cls.temp_dir.name, "malicious_traffic.csv")

        normal_df.to_csv(normal_file, index=True)
        malicious_df.to_csv(malicious_file, index=True)

        cls.csv_files = [normal_file, malicious_file]

        cls.generator = RealTrafficGenerator(
            num_satellites=6,
            num_orbits=2,
            satellites_per_orbit=3
        )

        cls.feature_dim, cls.num_classes = cls.generator.load_and_preprocess_data(
            cls.csv_files,
            test_size=0.2
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_load_and_preprocess_data(self):
        self.assertEqual(self.feature_dim, 20)
        self.assertEqual(self.num_classes, 2)
        self.assertTrue(hasattr(self.generator, 'X_train_tensor'))
        self.assertTrue(hasattr(self.generator, 'y_train_tensor'))
        self.assertTrue(hasattr(self.generator, 'X_test_tensor'))
        self.assertTrue(hasattr(self.generator, 'y_test_tensor'))
        self.assertTrue(hasattr(self.generator, 'label_encoder'))
        self.assertIn('normal', self.generator.get_class_names())
        self.assertIn('malicious', self.generator.get_class_names())
        self.assertTrue(hasattr(self.generator, 'scaler'))

    def test_generate_iid_data(self):
        satellite_datasets = self.generator.generate_data(iid=True)
        expected_satellites = 6
        self.assertEqual(len(satellite_datasets), expected_satellites)
        for orbit in range(1, 3):
            for sat in range(1, 4):
                self.assertIn(f"satellite_{orbit}-{sat}", satellite_datasets)
        for sat_id, dataset in satellite_datasets.items():
            self.assertIsInstance(dataset, TrafficFlowDataset)
            self.assertEqual(dataset.features.shape[1], self.feature_dim)

    def test_generate_non_iid_data(self):
        satellite_datasets = self.generator.generate_data(iid=False, alpha=0.1)
        self.assertGreater(len(satellite_datasets), 0)
        label_ratios = {}
        for sat_id, dataset in satellite_datasets.items():
            labels = dataset.labels.numpy()
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique, counts))
            total = sum(counts)
            if 0 in dist and 1 in dist:
                label_ratios[sat_id] = dist[0] / total
        if len(label_ratios) >= 2:
            ratios = list(label_ratios.values())
            self.assertGreater(max(ratios) - min(ratios), 0.1)

    def test_generate_test_data(self):
        test_dataset = self.generator.generate_test_data()
        self.assertIsInstance(test_dataset, TrafficFlowDataset)
        self.assertEqual(test_dataset.features.shape[1], self.feature_dim)

    def test_accessor_methods(self):
        self.assertEqual(self.generator.get_feature_dim(), self.feature_dim)
        self.assertEqual(self.generator.get_num_classes(), self.num_classes)
        class_names = self.generator.get_class_names()
        self.assertEqual(len(class_names), self.num_classes)
        self.assertIn('normal', class_names)
        self.assertIn('malicious', class_names)

if __name__ == "__main__":
    unittest.main()
