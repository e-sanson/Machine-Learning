import json
import os.path
import re
import ipykernel
import requests
import logging
import sys
import tensorflow.keras as keras
import datetime

from urllib.parse import urljoin

from notebook.notebookapp import list_running_servers


def get_logger(filename="notebook.log"):
    return open(filename,"a")

class ReportCallback(keras.callbacks.Callback):
        def __init__(self,frequency,use_val=True):
            self.file=log_file
            self.freq=frequency
            self.use_val=use_val
            self.separator=" || "
            if not(self.use_val):
                self.separator="\n"
        def on_epoch_end(self, epoch, logs={}):
            if (epoch % self.freq ==0):
                train_loss=logs["loss"]
                train_acc=logs["accuracy"]
                print(f"\t{epoch}: TRAIN loss {train_loss:.4f},  acc {train_acc:.4f}",end=self.separator)
                if self.use_val:
                    val_loss=logs["val_loss"]
                    val_acc=logs["val_accuracy"]
                    print(f"VAL loss {val_loss:.4f}, acc {val_acc:.4f}")


class LoggingCallback(keras.callbacks.Callback):
    def __init__(self,frequency,file,use_val=True):
        self.last_log=None
        self.last_epoch=None
        self.file=file
        self.freq=frequency
        self.use_val=use_val
        self.separator=" || "
        if not(self.use_val):
            self.separator="\n"
    def report(self,epoch,logs):
        date_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        msg=f"{date_str}|"
        train_loss=logs["loss"]
        train_acc=logs["accuracy"]
        msg+=f"\t{epoch}: TRAIN loss {train_loss:.4f},  acc {train_acc:.4f}"
        if self.use_val:
            val_loss=logs["val_loss"]
            val_acc=logs["val_accuracy"]         
            msg+=f" {self.separator} VAL loss {val_loss:.4f}, acc {val_acc:.4f}"
        print(msg)
        print(msg,file=self.file)
        self.file.flush()

    def on_epoch_end(self, epoch, logs={}):
        self.last_epoch=epoch
        self.last_log=logs
        if  (epoch % self.freq ==0):
            self.report(epoch,logs)
            self.last_epoch=None
            self.last_log=None
    # make sure we always report the final state after finishing training if we did not already
    def on_train_end(self,logs=None):
        if self.last_epoch:
            self.report(self.last_epoch,self.last_log)