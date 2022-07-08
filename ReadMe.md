# Action Classification and MediaPipe Integration Package

## Installation

### 1. Clone and build this package
```
cd ~/git
git clone git@github.com:akash1306/action_classification.git
```
```
cd ~/
mkdir john_doe_workspace
cd john_doe_workspace
catkin init
mkdir src
cd src
ln -s ~/git/action_classification
cd ..
catkin build
```

### 2. Install MediaPipe
Follow the steps on the official MediaPipe website [here](https://google.github.io/mediapipe/getting_started/install.html). 

*Note:*  There is currently no way to use MP without Bazel. 

### 3. TMUX and Tmuxinator
***TMUX - terminal multiplexer***

Tmux is a command-line utility that allows splitting a terminal to multiple panels and creating windows (tabs). It is similar to, e.g., Terminator, but runs entirely in the command line. Thus it can be used remotely over ssh. It is scriptable, which makes it ideal for automating processes, where multiple programs are launches simultaneously.

* https://github.com/tmux/tmux
* We compile tmux 3.0a from sources.
* List of basic key bindings for our particular setup can be found here: https://github.com/klaxalk/linux-setup/wiki/tmux
* The key bindings should be familiar to those using Vim.


***Tmuxinator - automating tmux***

Tmux itself is very powerful, tmuxinator is just adding some cream to it. Tmuxinator uses .xml files containing a description of a tmux session. It allows us to define and automate complex multi-terminal setups for, e.g., development (one session per program) and simulations. All our simulation startup script are written for tmuxinator.

* https://github.com/tmuxinator/tmuxinator

### 4. MRS System 
Please follow the instructions on the official repository of MRS for installation. 
* https://github.com/ctu-mrs/mrs_uav_system
* Documentation: https://ctu-mrs.github.io/

## To-Do

For example of mediapipe over ROS msg, see pose_treacking
For example of mediapipe on a video file, see keypoint extraction