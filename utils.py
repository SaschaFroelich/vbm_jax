#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:52:24 2023

@author: sascha
"""

import env
import numpy
import scipy
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import models_torch as models

from statistics import mean, stdev
from math import sqrt

def get_participant_data(file_day1, group, data_dir, published_results=0):
    "Get data of an individual participant"
    assert (group < 4)
    "RETURN: data (dict) used for inference"

    if published_results:
        ID = file_day1.split("/")[-1][4:9]
    else:
        ID = file_day1.split("/")[-1][4:28]  # Prolific ID

    print(data_dir)
    print(glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat" % (group+1, ID)))

    file_day2 = glob.glob(
        data_dir + "Grp%d/csv/*%s*Tag2*.mat" % (group+1, ID))[0]

    print("==============================================================")
    print("Doing %s and %s" % (file_day1, file_day2))
    participant_day1 = scipy.io.loadmat(file_day1)
    participant_day2 = scipy.io.loadmat(file_day2)

    correct = []  # 0 = wrong response, 1 = correct response, 2 = too slow, 3 = two keys at once during joker-trials
    choices = []
    outcomes = []
    RT = []

    "Block order is switched pairwise for groups 2 & 4"
    block_order_day1 = [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4], [
        0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]]

    for i in block_order_day1[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2)  # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)

        correct.extend(numpy.squeeze(
            participant_day1["correct_all_cell"][0][i]).tolist())
        choices.extend(numpy.squeeze(
            participant_day1["resps_response_digit_cell"][0][i]).tolist())  # Still 1-indexed
        outcomes.extend(numpy.squeeze(
            participant_day1["rew_cell"][0][i]).tolist())
        RT.extend(numpy.squeeze(participant_day1["RT_cell"][0][i]).tolist())

    block_order_day2 = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6], [
        0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]]

    for i in block_order_day2[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2)  # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)

        correct.extend(numpy.squeeze(
            participant_day2["correct_all_cell"][0][i]).tolist())
        choices.extend(numpy.squeeze(
            participant_day2["resps_response_digit_cell"][0][i]).tolist())  # Still 1-indexed
        outcomes.extend(numpy.squeeze(
            participant_day2["rew_cell"][0][i]).tolist())
        RT.extend(numpy.squeeze(participant_day2["RT_cell"][0][i]).tolist())

    "Errors to -9 (0 -> -9)"
    choices = [ch if ch != 0 else -9 for ch in choices]

    "Transform choices to 0-indexing by subtracting 1 -> this transforms errors to -10, leave -1 (new block) unchanged"
    # Now new block is again -1
    choices = [ch-1 if ch != -1 else -1 for ch in choices]

    "Transform outcomes: 2 (no reward) -> 0, -1 (error) -> -10, 1 -> 1, -2->-1"
    outcomes = [0 if out == 2 else -10 if out == -
                1 else 1 if out == 1 else -1 for out in outcomes]

    "Check internal consistency: indexes of errors should be the same in choices and in correct"
    indices_ch = [i for i, x in enumerate(choices) if x == -10]
    indices_corr = [i for i, x in enumerate(correct) if x != 1 and x != -1]

    assert (indices_ch == indices_corr)

    "Check internal consistency: indexes of new blocks (-1) should be the same in choices and in correct"
    indices_ch = [i for i, x in enumerate(choices) if x == -1]
    indices_out = [i for i, x in enumerate(outcomes) if x == -1]
    indices_corr = [i for i, x in enumerate(correct) if x == -1]

    assert (indices_ch == indices_corr)
    assert (indices_ch == indices_out)

    trialsequence = []
    trialsequence_wo_jokers = []
    blocktype = []
    blockidx = []
    for block in range(14):
        "Mark the beginning of a new block"
        trialsequence.append(torch.tensor(-1))
        trialsequence_wo_jokers.append(torch.tensor(-1))
        blocktype.append("n")
        blockidx.append(block)

        seq, btype, seq_wo_jokers = get_trialseq(
            group, block, published_results=published_results)

        trialsequence.extend(seq)
        trialsequence_wo_jokers.extend(seq_wo_jokers)
        blocktype.extend([btype]*len(seq))
        blockidx.extend([block]*len(seq))

    assert (len(trialsequence) == len(choices))
    assert (len(outcomes) == len(choices))
    assert (len(outcomes) == len(blocktype))

    trialsequence = [[trialsequence[i].item()]
                     for i in range(len(trialsequence))]
    trialsequence_wo_jokers = [[trialsequence_wo_jokers[i].item()] for i in range(
        len(trialsequence_wo_jokers))]
    choices = [torch.tensor([choices[i]]) for i in range(len(choices))]
    outcomes = [torch.tensor([outcomes[i]]) for i in range(len(outcomes))]
    blocktype = [[blocktype[i]] for i in range(len(blocktype))]
    blockidx = [[blockidx[i]] for i in range(len(blockidx))]
    RT = [[RT[i]] for i in range(len(RT))]

    data = {"Trialsequence": trialsequence,
            "Trialsequence no jokers": trialsequence_wo_jokers,
            "Choices": choices,
            "Outcomes": outcomes,
            "Blocktype": blocktype,
            "Blockidx": blockidx,
            "RT": RT}

    return data, ID


def get_trialseq(group, block_no, published_results=0):
    '''
    NB: in mat-files, the block order was already swapped, as if all
    participants saw the first group's block order! Have to correct for this!
    '''

    "This is the blockorder participants actually saw"
    blockorder = [["random1", "trainblock1", "random2", "trainblock2",
                  "random3", "trainblock3", "trainblock4", "random4",
                  "trainblock5", "random5", "trainblock6", "random6",
                  "trainblock7", "random7"], ["trainblock1", "random1",
                  "trainblock2", "random2", "trainblock3", "random3",
                  "random4", "trainblock4", "random5", "trainblock5",
                  "random6", "trainblock6", "random7", "trainblock7"],
                  ["mirror_random1", "mirror_trainblock1", "mirror_random2",
                  "mirror_trainblock2", "mirror_random3", "mirror_trainblock3",
                  "mirror_trainblock4", "mirror_random4", "mirror_trainblock5",
                  "mirror_random5", "mirror_trainblock6", "mirror_random6",
                  "mirror_trainblock7", "mirror_random7"],
                  ["mirror_trainblock1", "mirror_random1",
                  "mirror_trainblock2", "mirror_random2", "mirror_trainblock3",
                  "mirror_random3", "mirror_random4", "mirror_trainblock4",
                  "mirror_random5", "mirror_trainblock5", "mirror_random6",
                  "mirror_trainblock6", "mirror_random7",
                  "mirror_trainblock7"]]

    if published_results:
        "Published"
            mat = scipy.io.loadmat(
                "/home/sascha/Desktop/vb_model/vbm_torch/matlabcode/published/%s.mat" % blockorder[group][block_no])

    else:
        "Clipre"
            mat = scipy.io.loadmat(
                "/home/sascha/Desktop/vb_model/vbm_torch/matlabcode/clipre/%s.mat" % blockorder[group][block_no])

    types = [["r", "s", "r", "s", "r", "s", "s", "r", "s", "r", "s", "r", "s", "r"],
             ["s", "r", "s", "r", "s", "r", "r", "s", "r", "s", "r", "s", "r", "s"],
             ["r", "s", "r", "s", "r", "s", "s", "r", "s", "r", "s", "r", "s", "r"],
             ["s", "r", "s", "r", "s", "r", "r", "s", "r", "s", "r", "s", "r", "s"]]

    return torch.tensor(numpy.squeeze(mat["sequence"])), types[group][block_no], \
        torch.tensor(numpy.squeeze(mat["sequence_without_jokers"]))


def replace_single_element_lists(value):
    "Replace a list named value with its first entry"
    import torch

    if not isinstance(value, int) and len(value) == 1:
        if torch.is_tensor(value[0]):
            return value[0].item()
        else:
            return value[0]
    else:
        return value


def arrange_data_for_plot(i, df, **kwargs):
    "lists -> values"
    df = df.applymap(replace_single_element_lists)

    "Only retain rows pertaining to joker trial"
    df = df[df["Jokertypes"] > -1]

    "Get rid of tensors"
    df = df.applymap(lambda x: x.item() if torch.is_tensor(x) else x)

    """
    Choices -> Goal-Directed Choices.
    0: Low-Reward Choice, 1: High-Reward Choice 
    groups 0 & 1: (from High-Rew: 0 & 3, Low-Rew: 1 & 2)
    groups 2 & 3: (from High-Rew: 1 & 2, Low-Rew: 0 & 3)
    """

    if kwargs["group"] == 0 or kwargs["group"] == 1:
        df['Choices'] = df['Choices'].map(
            lambda x: 1 if x == 0 else 0 if x == 1 else 0 if x == 2 else 1)

    elif kwargs["group"] == 2 or kwargs["group"] == 3:
        df['Choices'] = df['Choices'].map(
            lambda x: 0 if x == 0 else 1 if x == 1 else 1 if x == 2 else 0)

    else:
        raise Exception("Group is not correctly specified, pal!")

    data_new = {"HPCF": [], "Trialtype": [], "Blockidx": [], "datatype": []}
    data_Q = {"Qdiff": [], "Blocktype": [], "Blockidx": []}

    for block in df["Blockidx"].unique():
        # For column "Jokertypes":
        # -1 no joker
        # 0 random
        # 1 congruent
        # 2 incongruent

        if "Qdiff" in df.columns:
            data_Q["Qdiff"].append(df[df["Blockidx"] == block]["Qdiff"].mean())

        if df[df["Blockidx"] == block]["Blocktype"].unique()[0] == "r":
            "Random Block"

            if "Qdiff" in df.columns:
                data_Q["Blocktype"].append("r")
                data_Q["Blockidx"].append(block)

            data_new["Blockidx"].append(block)
            data_new["Trialtype"].append("random")
            data_new["HPCF"].append(df[(df["Blockidx"] == block) & (
                df["Jokertypes"] == 0)]["Choices"].mean())

            if i == 0:
                data_new["datatype"].append(
                    "given (Group %s)" % kwargs["group"])

            elif i == 1:
                data_new["datatype"].append(
                    "simulated (Group %s)" % kwargs["group"])

            else:
                raise Exception("Fehla!")

        elif df[df["Blockidx"] == block]["Blocktype"].unique()[0] == "s":
            "Sequential Block"

            if "Qdiff" in df.columns:
                data_Q["Blocktype"].append("s")
                data_Q["Blockidx"].append(block)

            "Congruent Jokers"
            data_new["Blockidx"].append(block)
            data_new["Trialtype"].append("congruent")
            data_new["HPCF"].append(df[(df["Blockidx"] == block) & (
                df["Jokertypes"] == 1)]["Choices"].mean())

            if i == 0:
                data_new["datatype"].append(
                    "given (Group %s)" % kwargs["group"])

            elif i == 1:
                data_new["datatype"].append(
                    "simulated (Group %s)" % kwargs["group"])

            else:
                raise Exception("Fehla!")

            "Incongruent Jokers"
            data_new["Blockidx"].append(block)
            data_new["Trialtype"].append("incongruent")
            data_new["HPCF"].append(df[(df["Blockidx"] == block) & (
                df["Jokertypes"] == 2)]["Choices"].mean())

            if i == 0:
                data_new["datatype"].append(
                    "given (Group %s)" % kwargs["group"])

            elif i == 1:
                data_new["datatype"].append(
                    "simulated (Group %s)" % kwargs["group"])

            else:
                raise Exception("Fehla!")

        else:
            raise Exception("Fehla!")

    return data_new, data_Q


def plot_results(data_sim, *args, **kwargs):
    """
    data_sim :  DataFrame with data of single participant to be plotted.
    *args :     - DataFrame with second set of data to be plotted (for comparison with first DataFrame)
    **kwargs :  group

    Jokertypes : -1 no joker, 0 random , 1 congruent, 2 incongruent
    """

    if args:
        datas = (data_sim, args[0])

    elif not args:
        datas = (data_sim,)

    else:
        raise Exception("Fehla!")

    "Create df_new, which will contain modified results for plotting and will contain column 'datatype' for simulated data and experimental data"
    df_new = pd.DataFrame()

    for i in range(len(datas)):

        # print(i)
        df = pd.DataFrame(data=datas[i])

        data_new, data_Q = arrange_data_for_plot(i, df, **kwargs)

        df_temp = pd.DataFrame(data=data_new)
        df_new = pd.concat([df_new, df_temp])
        df_Q = pd.DataFrame(data=data_Q)

    # fig, ax = plt.subplots()
    if kwargs["group"] == 0 or kwargs["group"] == 2:
        custom_palette = ['r', 'g', 'b']  # random, congruent, incongruent

    elif kwargs["group"] == 1 or kwargs["group"] == 3:
        custom_palette = ['g', 'b', 'r']  # congruent, incongruent, random

    sns.relplot(x="Blockidx", y="HPCF", hue="Trialtype", data=df_new,
                kind="line", col="datatype", palette=custom_palette)
    # plt.plot([5.5, 5.5],[0.5,1], color='black') # plot Day1/Day2 line

    "----- UNCOMMENT THESE TWO LINES TO PLOT QDIFF ------------"
    # if "Qdiff" in df.columns:
    #     sns.relplot(x='Blockidx', y="Qdiff", hue ="Blocktype", data=df_Q, kind = 'line')
    "----- --------------------------------------- ------------"

    if "ymin" in kwargs:
        ylim = kwargs["ymin"]
    else:
        ylim = 0

    plt.ylim([ylim, 1])

    if "omega_true" in kwargs:
        plt.text(0.05, ylim+0.14, "omega = %.2f, inf:%.2f" %
                 (kwargs["omega_true"], kwargs["omega_inf"]))
        plt.text(0.05, ylim+0.08, "dectemp = %.2f, inf:%.2f" %
                 (kwargs["dectemp_true"], kwargs["dectemp_inf"]))
        plt.text(0.05, ylim+0.02, "lr = %.2f, inf:%.2f" %
                 (kwargs["lr_true"], kwargs["lr_inf"]))

    else:
        if "omega_inf" in kwargs:
            plt.text(0.05, ylim+0.14, "omega inf:%.2f" % (kwargs["omega_inf"]))
            plt.text(0.05, ylim+0.08, "dectemp inf:%.2f" %
                     (kwargs["dectemp_inf"]))
            plt.text(0.05, ylim+0.02, "lr inf:%.2f" % (kwargs["lr_inf"]))

    if "savedir" in kwargs:

        if "plotname" in kwargs:
            plt.savefig(kwargs["savedir"]+"/%s.png" % kwargs["plotname"])
        else:
            plt.savefig(kwargs["savedir"]+"/plot_%d.png" %
                        (numpy.random.randint(10e+9)))

    plt.show()

    return df_new[df_new["datatype"] == "simulated (Group %d)" % kwargs["group"]], \
        df_new[df_new["datatype"] == "given (Group %d)" % kwargs["group"]]


def cohens_d(c0, c1):
    return (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))


def simulate_model_behaviour(num_agents, model, **kwargs):

    assert (model == 'B')

    if 'k' in kwargs:
        k = kwargs['k']

    else:
        k = 4.

    df_all = pd.DataFrame()

    for agent in range(num_agents):
        print("Simulating agent no. %d" % agent)
        newagent = models.Vbm_B(theta_rep_day1=kwargs['theta_rep_day1'],
                                theta_rep_day2=kwargs['theta_rep_day2'],
                                lr_day1=kwargs['lr_day1'],
                                lr_day2=kwargs['lr_day2'],
                                theta_Q_day1=kwargs['theta_Q_day1'],
                                theta_Q_day2=kwargs['theta_Q_day2'],
                                k=k,
                                Q_init=[0.2, 0., 0., 0.2])

        newenv = env.env(newagent, rewprobs=[
                         0.8, 0.2, 0.2, 0.8], matfile_dir='./matlabcode/clipre/')

        newenv.run()
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"],
                "Qdiff": [(newenv.agent.Q[i][..., 0] + newenv.agent.Q[i][..., 3])/2 - (newenv.agent.Q[i][..., 1] + newenv.agent.Q[i][..., 2])/2 for i in range(len(newenv.choices))]}

        df = pd.DataFrame(data)

        data_new, data_Q = arrange_data_for_plot(1, df, group=0)

        df_new = pd.DataFrame(data_new)

        df_all = pd.concat((df_all, df_new))

    custom_palette = ['r', 'g', 'b']  # random, congruent, incongruent
    sns.relplot(x="Blockidx", y="HPCF", hue="Trialtype",
                data=df_all, kind="line", palette=custom_palette)
    plt.plot([5.5, 5.5], [0.5, 1], color='black')
    plt.show()


def comp_groupdata(groupdata, for_ddm=1):
    """Trialsequence no jokers and RT are only important for DDM to determine the jokertype"""

    if for_ddm:
        newgroupdata = {"Trialsequence": [],\
                        # "Trialsequence no jokers" : [],\
                        "Choices": [],\
                        "Outcomes": [],\
                        "Blocktype": [],\
                        "Blockidx": [],\
                        "RT": []}

    else:
        newgroupdata = {"Trialsequence": [],\
                        # "Trialsequence no jokers" : [],\
                        "Choices": [],\
                        "Outcomes": [],\
                        "Blocktype": [],\
                        "Blockidx": []}

    for trial in range(len(groupdata[0]["Trialsequence"])):
        trialsequence = []
        # trialseq_no_jokers = []
        choices = []
        outcomes = []
        blocktype = []
        blockidx = []
        if for_ddm:
            RT = []

        for dt in groupdata:
            trialsequence.append(dt["Trialsequence"][trial][0])
            # trialseq_no_jokers.append(dt["Trialsequence no jokers"][trial][0])
            choices.append(dt["Choices"][trial][0].item())
            outcomes.append(dt["Outcomes"][trial][0].item())
            blocktype.append(dt["Blocktype"][trial][0])
            blockidx.append(dt["Blockidx"][trial][0])
            if for_ddm:
                RT.append(dt["RT"][trial][0])

        newgroupdata["Trialsequence"].append(trialsequence)
        # newgroupdata["Trialsequence no jokers"].append(trialseq_no_jokers)
        newgroupdata["Choices"].append(torch.tensor(choices, dtype=int))
        newgroupdata["Outcomes"].append(torch.tensor(outcomes, dtype=int))
        newgroupdata["Blocktype"].append(blocktype)
        newgroupdata["Blockidx"].append(blockidx)
        if for_ddm:
            newgroupdata["RT"].append(RT)

    return newgroupdata
