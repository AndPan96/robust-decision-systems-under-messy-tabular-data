from unittest.mock import patch
import logging
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler
from src.config.paths import LOGS_FILE, RAW_APPLICATION_TRAIN, UPDATES_FILE
from src.pipelines.inference import inference_pipeline
from src.pipelines.monitoring import monitoring_pipeline
from src.pipelines.retrain_all import retrain_all_pipeline
from src.pipelines.retrain_single import retrain_single_pipeline
from src.pipelines.up_rows import update_rows_pipeline
import json
from src.models.linear_model import LinearModel
from src.models.mlp import MLP
import time

from src.config.state import load_state

# same logging configuration as scheduler.py
handler = RotatingFileHandler(LOGS_FILE,maxBytes=5_000_000,backupCount=3)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[handler])

logger = logging.getLogger(__name__)

test_registry = {
            "LinearModel": {
                "class": LinearModel,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": .01, "batch_size": 256, "steps": 1}
            },
            "MLP": {
                "class": MLP,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": .01, "batch_size": 256, "steps": 1}
            }
        }

def test_operations(set_environment, reset_environment):
 
     try:
         set_environment()
 
         df = pd.read_csv(RAW_APPLICATION_TRAIN)
         sample = df.iloc[:12000].copy()
         df.iloc[12000:].to_csv(RAW_APPLICATION_TRAIN, index=False)
 
         feature_cols = [
             c for c in df.columns
             if c not in ["SK_ID_CURR", "TARGET"]
             and not any(c.startswith(p) for p in ["B_", "BB_", "PA_", "PCB_", "IP_", "CCB_"])
         ]
         feature_col = feature_cols[0]
 
         phase1 = sample.copy()
         phase1["TARGET"] = None
         phase1[feature_col] = np.nan
         updates_phase1 = {
             "application_train": phase1.to_dict(orient="records")
         }
 
         b1 = sample.iloc[0:3000].copy()
         b2 = sample.iloc[3000:6000].copy()
         b3 = sample.iloc[6000:9000].copy()
         b4 = sample.iloc[9000:12000].copy()
 
         b1["TARGET"] = None
         b1[feature_col] = np.nan
         updates_b1 = {
             "application_train": b1.to_dict(orient="records")
         }
 
         b2[feature_col] = np.nan
         updates_b2 = {
             "application_train": b2.to_dict(orient="records")
         }
 
         updates_b3 = {
             "application_train": b3.to_dict(orient="records")
         }
 
         updates_b4 = {
             "application_train": b4.to_dict(orient="records")
         }
 
         # ----------
 
         # (check "retrain_required" True)
         assert load_state()["retrain_required"] == True
 
         # first training
         with patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
             patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
             retrain_all_pipeline()
 
         # (check "retrain_required" False)
         assert load_state()["retrain_required"] == False
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # inference with no unlabeled data
         inference_pipeline()
 
         # update adding unlabeled rows
         with open(UPDATES_FILE, "w") as f:
             json.dump(updates_phase1, f)
         update_rows_pipeline()
         # (check "transform_required" True)
         assert load_state()["transform_required"] == True
 
         # inference on unlabeled data
         inference_pipeline()
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # update without target
         with open(UPDATES_FILE, "w") as f:
             json.dump(updates_b1, f)
         update_rows_pipeline()
         # (check "transform_required" True)
         assert load_state()["transform_required"] == True
 
         # try inference with updated unlabeled data
         inference_pipeline()
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # try monitoring with |target processed rows| = 0
         monitoring_pipeline()
 
         # ----------
 
         # update with |target processed rows| < MONITORING_WINDOW 
         with open(UPDATES_FILE, "w") as f:
             json.dump(updates_b2, f)
         update_rows_pipeline()
         # (check "transform_required" True)
         assert load_state()["transform_required"] == True
 
         # try inference but updated rows are sealed
         inference_pipeline()
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # try monitoring with no data enough
         monitoring_pipeline()
 
         # ----------
 
         # update with |target processed rows| > MONITORING_WINDOW
         with open(UPDATES_FILE, "w") as f:
             json.dump(updates_b3, f)
         update_rows_pipeline()
         # (check "transform_required" True)
         assert load_state()["transform_required"] == True
 
         # try inference but updated rows are sealed
         inference_pipeline()
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # monitoring with enough data (monitoring report should be created)
         monitoring_pipeline()
 
         # ----------
 
         # 2nd update with |target processed rows| > MONITORING_WINDOW
         with open(UPDATES_FILE, "w") as f:
             json.dump(updates_b4, f)
         update_rows_pipeline()
         # (check "transform_required" True)
         assert load_state()["transform_required"] == True
 
         # try inference but updated rows are sealed
         inference_pipeline()
         # (check "transform_required" False)
         assert load_state()["transform_required"] == False
 
         # monitoring with enough data (monitoring report should update)
         monitoring_pipeline()
 
     finally:
        reset_environment()



def test_retraining_no_need(set_environment, reset_environment):

    try:
        set_environment()

        df = pd.read_csv(RAW_APPLICATION_TRAIN)
        sample = df.iloc[:6000].copy()
        df.iloc[6000:].to_csv(RAW_APPLICATION_TRAIN, index=False)
        updates_sample = {
            "application_train": sample.to_dict(orient="records")
        }

        feature_cols = [
            c for c in df.columns
            if c not in ["SK_ID_CURR", "TARGET"]
            and not any(c.startswith(p) for p in ["B_", "BB_", "PA_", "PCB_", "IP_", "CCB_"])
        ]
        feature_col = feature_cols[0]

        phase1 = sample.copy()
        phase1["TARGET"] = None
        phase1[feature_col] = None
        updates_phase1 = {
            "application_train": phase1.to_dict(orient="records")
        }

        # ----------

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True

        # first training
        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update adding unlabeled rows
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_phase1, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # inference on unlabeled data
        inference_pipeline()
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update with |target processed rows| > MONITORING_WINDOW
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_sample, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # ----------

        # simulate monitoring below threshold (force retraining single)
        with patch("src.config.data_config.RETRAIN_THRESHOLD", 0.01):
            monitoring_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False

    finally:
        reset_environment()



def test_retraining_single_ok(set_environment, reset_environment):

    try:
        set_environment()

        df = pd.read_csv(RAW_APPLICATION_TRAIN)
        sample = df.iloc[:6000].copy()
        df.iloc[6000:].to_csv(RAW_APPLICATION_TRAIN, index=False)
        updates_sample = {
            "application_train": sample.to_dict(orient="records")
        }

        feature_cols = [
            c for c in df.columns
            if c not in ["SK_ID_CURR", "TARGET"]
            and not any(c.startswith(p) for p in ["B_", "BB_", "PA_", "PCB_", "IP_", "CCB_"])
        ]
        feature_col = feature_cols[0]

        phase1 = sample.copy()
        phase1["TARGET"] = None
        phase1[feature_col] = None
        updates_phase1 = {
            "application_train": phase1.to_dict(orient="records")
        }

        # ----------

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True

        # first training
        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update adding unlabeled rows
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_phase1, f)
        update_rows_pipeline()   
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # inference on unlabeled data
        inference_pipeline()
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update with |target processed rows| > MONITORING_WINDOW
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_sample, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # ----------

        # simulate monitoring below threshold (force retraining single)
        with patch("src.actions.monitor_model.RETRAIN_THRESHOLD", 0.99):
            monitoring_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False

        # ----------

        # simulate model above threshold 
        with patch("src.actions.test_deploy_model.DEPLOY_THRESHOLD", 0.01), \
            patch("src.actions.train_model.MODEL_REGISTRY", test_registry):
            retrain_single_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False


    finally:
        reset_environment()



def test_retraining_single_nok_all_ok(set_environment, reset_environment):

    try:
        set_environment()

        df = pd.read_csv(RAW_APPLICATION_TRAIN)
        sample = df.iloc[:6000].copy()
        df.iloc[6000:].to_csv(RAW_APPLICATION_TRAIN, index=False)
        updates_sample = {
            "application_train": sample.to_dict(orient="records")
        }

        feature_cols = [
            c for c in df.columns
            if c not in ["SK_ID_CURR", "TARGET"]
            and not any(c.startswith(p) for p in ["B_", "BB_", "PA_", "PCB_", "IP_", "CCB_"])
        ]
        feature_col = feature_cols[0]

        phase1 = sample.copy()
        phase1["TARGET"] = None
        phase1[feature_col] = None
        updates_phase1 = {
            "application_train": phase1.to_dict(orient="records")
        }

        # ----------

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True

        # first training
        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()
            
        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update adding unlabeled rows
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_phase1, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # inference on unlabeled data
        inference_pipeline()
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update with |target processed rows| > MONITORING_WINDOW
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_sample, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # ----------

        # simulate monitoring below threshold (force retraining single)
        with patch("src.actions.monitor_model.RETRAIN_THRESHOLD", 0.99):
            monitoring_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False

        # ----------

        # simulate model below threshold (force retraining all)
        with patch("src.actions.test_deploy_model.DEPLOY_THRESHOLD", 0.99), \
            patch("src.actions.train_model.MODEL_REGISTRY", test_registry):
            retrain_single_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" True)
        assert load_state()["retrain_single"] == True

        # ----------
        with patch("src.actions.test_deploy_all.DEPLOY_THRESHOLD", 0.01), \
            patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False
        # (check "retrain_all" False)
        assert load_state()["retrain_all"] == False

    finally:
        reset_environment()



def test_retraining_all_nok_email_ok(set_environment, reset_environment):

    try:
        set_environment()

        df = pd.read_csv(RAW_APPLICATION_TRAIN)
        sample = df.iloc[:6000].copy()
        df.iloc[6000:].to_csv(RAW_APPLICATION_TRAIN, index=False)
        updates_sample = {
            "application_train": sample.to_dict(orient="records")
        }

        feature_cols = [
            c for c in df.columns
            if c not in ["SK_ID_CURR", "TARGET"]
            and not any(c.startswith(p) for p in ["B_", "BB_", "PA_", "PCB_", "IP_", "CCB_"])
        ]
        feature_col = feature_cols[0]

        phase1 = sample.copy()
        phase1["TARGET"] = None
        phase1[feature_col] = None
        updates_phase1 = {
            "application_train": phase1.to_dict(orient="records")
        }

        # ----------

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True

        # first training
        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

        # (check "retrain_required" False)
        assert load_state()["retrain_required"] == False
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update adding unlabeled rows
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_phase1, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # inference on unlabeled data
        inference_pipeline()
        # (check "transform_required" False)
        assert load_state()["transform_required"] == False

        # update with |target processed rows| > MONITORING_WINDOW
        with open(UPDATES_FILE, "w") as f:
            json.dump(updates_sample, f)
        update_rows_pipeline()
        # (check "transform_required" True)
        assert load_state()["transform_required"] == True

        # ----------

        # simulate monitoring below threshold (force retraining single)
        with patch("src.actions.monitor_model.RETRAIN_THRESHOLD", 0.99):
            monitoring_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" False)
        assert load_state()["retrain_single"] == False

        # ----------

        # simulate model below threshold (force retraining all)
        with patch("src.actions.test_deploy_model.DEPLOY_THRESHOLD", 0.99), \
            patch("src.actions.train_model.MODEL_REGISTRY", test_registry):
            retrain_single_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" True)
        assert load_state()["retrain_single"] == True
        # (check "retrain_all" False)
        assert load_state()["retrain_all"] == False

        # ----------

        # simulate model below threshold (force email submission)
        with patch("src.actions.test_deploy_all.DEPLOY_THRESHOLD", 0.99), \
            patch("src.actions.test_deploy_all.send_email") as mock_email, \
            patch("src.actions.train_model.MODEL_REGISTRY", test_registry), \
            patch("src.actions.train_all.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

        # (check "retrain_required" True)
        assert load_state()["retrain_required"] == True
        # (check "retrain_single" True)
        assert load_state()["retrain_single"] == True
        # (check "retrain_all" True)
        assert load_state()["retrain_all"] == True
        # check email sent
        mock_email.assert_called_once()

    finally:
        reset_environment()