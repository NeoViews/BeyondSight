"""
tracklets_visualisation.py
Author: BeyondInsight Team
Date: Nov 19, 2024

Description:
This script visualizes the tracking data of players and the ball on a soccer pitch. 
It uses Matplotlib to plot positions frame-by-frame, enabling analysis of player 
and ball movement dynamics during the game.

Key Features:
- Functionality to plot a soccer pitch with customizable dimensions and style.
- Visualization of player and ball positions across frames.
- Frame-by-frame animation of the soccer match.

Usage:
Run this script to generate a visual representation of the soccer match. 
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_pitch_v2(
    figax=None,
    field_dimen=(106.0, 68.0),
    field_color="green",
    linewidth=2,
    markersize=20,
):
    """plot_pitch

    Plots a soccer pitch. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    if figax is None:  # create new pitch
        fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
    else:  # overlay on a previously generated pitch
        fig, ax = figax  # unpack tuple
    # fig,ax = plt.subplots(figsize=(12,8)) # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == "green":
        # if field_color=='black':
        ax.set_facecolor("mediumseagreen")
        # ax.set_facecolor('black')
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    elif field_color == "white":
        lc = "k"
        pc = "k"
    # ALL DIMENSIONS IN m
    # border_dimen = (3,3) # include a border arround of the field of width 3m
    border_dimen = (0, 0)  # include a border arround of the field of width 3m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch
    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard
    # plot half way line # center circle

    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize)
    y = np.linspace(-1, 1, 50) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth)
    ax.plot(-x, y, lc, linewidth=linewidth)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth,
        )
        # goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth,
        )
        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth,
        )
        # penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize,
        )
        # corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x,
            -half_pitch_width + y,
            lc,
            linewidth=linewidth,
        )
        ax.plot(
            s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth
        )
        # draw the D
        y = (
            np.linspace(-1, 1, 50) * D_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax


game_id = 2  # let's look at sample match 2

plt.ion()

nP_h = 14
nP_a = 12

dst_folder = "./data"


isDrawing = True
if isDrawing:
    plt.ion()
    field_dimen = (106.0, 68.0)
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]

    fig3 = plt.figure(figsize=(10.5, 6.8))
    axResult = fig3.add_subplot(111, aspect="equal")
    axResult.set_xlim([-xmax, xmax])
    axResult.set_ylim([-ymax, ymax])
    fig3, axResult = plot_pitch_v2((fig3, axResult))

colours = np.random.rand(100, 3)  # used only for display

colors2 = np.array(
    [
        "green",
        "green",
        "brown",
        "darkorange",
        "darkorange",
        "lightgreen",
        "lightgreen",
        "lightbrown",
        "yellow",
        "yellow",
        "magenta",
        "white",
        "orange",
        "purple",
        "beige",
        "gray",
        "brown",
        "cyan",
    ]
)

nb = 14
for s in range(2, nb):
    tracking_data_partial_filename = (
        dst_folder + "/tracking_data_partially_observed_subset" + str(s + 1) + ".csv"
    )
    tracking_data_partial = np.loadtxt(
        tracking_data_partial_filename, delimiter=",", skiprows=1
    )
    # tracking_data_full_filename = dst_folder+"/tracking_data_full_subset"+str(s+1)+".csv";
    # tracking_data_full = np.loadtxt(tracking_data_full_filename, delimiter=",")

    nrows = tracking_data_partial.shape[0]

    for j in range(1, nrows, 1):
        print("j=", j)
        if isDrawing:
            figobjs_results = []
            objs = axResult.annotate(str(j), (-52, -32), fontsize=12)
            figobjs_results.append(objs)

        c = tracking_data_partial[j, 0:2]  # ball
        if not np.isnan(c[0]):
            if isDrawing:
                objs = axResult.scatter(c[0], c[1], c="white", s=20)
                figobjs_results.append(objs)
            last_ball_pos = c

        camera_pos = tracking_data_partial[j, -8:]
        if not np.isnan(camera_pos[0]):
            wPs2 = camera_pos.reshape((4, 2)).transpose()
            wPs2 = np.hstack(
                (wPs2, wPs2[:, 0:1])
            )  # repeat the first point to create a 'closed loop'

            if isDrawing:
                (objs,) = axResult.plot(wPs2[0, :], -wPs2[1, :], "b")

                figobjs_results.append(objs)

        for idx in range(1, nP_h + nP_a + 1):
            if not np.isnan(tracking_data_partial[j, 2 * idx]):
                if idx == 1:
                    team = 0  # GK1
                elif idx < nP_a + 1:
                    team = 1
                    # team 1 player
                elif idx == nP_a + 1:
                    team = 4
                    # GK2
                else:
                    team = 3
                    # team 2 player

                c = tracking_data_partial[j, 2 * idx : 2 * idx + 2]
                color = colors2[team]
                """
                if within camera view: # for full tracking data
                    color = colors2[team]
                else:
                    color = colors2[team+5]
                """
                if isDrawing:
                    # if iP[0] >= 0 and iP[0] < Lx and iP[1] >= 0 and iP[1] <= Ly:
                    objs = axResult.scatter(c[0], c[1], c=color, s=40)
                    # objs = axResult.scatter(c[0], c[1], c=colors2[team],s=40)
                    figobjs_results.append(objs)
                    if True:  # False:##role is not None:
                        objs = axResult.annotate(
                            "GK" if (idx == 12 or idx == 26) else str(idx),
                            (c[0] - 0.5, c[1] - 0.5),
                            fontsize=8,
                        )
                        figobjs_results.append(objs)

        if isDrawing:
            fig3.canvas.flush_events()
            plt.draw()

        if isDrawing:
            xxx = 1
            for jjj, figobj in enumerate(figobjs_results):
                figobj.remove()
                xxx = 1
