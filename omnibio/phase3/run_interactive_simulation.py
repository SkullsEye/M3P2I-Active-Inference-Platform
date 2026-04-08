import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.behavior_tree import Blackboard, Status
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost, rectangle_obstacle_cost, escape_local_minimum_cost
from common.active_inference_planner import ActiveInferencePlanner
from common import plotting_utils

class SimulationState:
    def __init__(self):
        self.reset()

    def reset(self):
        print("Resetting simulation state...")
        self.initial_state = np.array([0.0, 0.0]); self.goal = np.array([10.0, 10.0])
        self.circ_obstacles = [np.array([5.0, 5.0])]; self.rect_obstacles = []
        self.ai_planner = ActiveInferencePlanner(goal=self.goal, initial_state=self.initial_state, num_rollouts=100, noise_level=2.5)
        self.normal_cost_functions = {
            'distance': (distance_to_goal_cost, 1.0),
            'circ_obstacle': (lambda s,a,g: obstacle_avoidance_cost(s, a, g, self.circ_obstacles), 1.0),
            'rect_obstacle': (lambda s,a,g: rectangle_obstacle_cost(s, a, g, self.rect_obstacles), 1.0),
            'control': (control_effort_cost, 0.1)
        }
        self.blackboard = Blackboard(); self.blackboard.set("state", self.initial_state)
        self.path_history = deque([self.initial_state.copy()], maxlen=1000)
        self.cost_history = []
        self.full_rollout_history = []
        self.recent_positions = deque(maxlen=10)
        self.stuck_counter, self.is_stuck = 0, False
        self.status, self.is_paused = Status.RUNNING, True
        self.placement_mode, self.final_rollouts_drawn = None, False

sim_state = SimulationState()
fig = plt.figure(figsize=(16, 9)); gs = gridspec.GridSpec(2, 3, figure=fig); plt.subplots_adjust(bottom=0.2)
ax_nav = fig.add_subplot(gs[:, 0:2]); ax_cost = fig.add_subplot(gs[0, 2]); ax_text = fig.add_subplot(gs[1, 2])
robot_path, robot_marker, best_rollout_line = None, None, None
cost_lines, hist_rollout_lines, current_rollout_lines, circ_patches, rect_patches = [], [], [], [], []

def setup_animated_artists():
    global robot_path, robot_marker, cost_lines, hist_rollout_lines, current_rollout_lines, best_rollout_line
    robot_path, = ax_nav.plot([], [], 'b-', linewidth=2, label='Robot Path')
    robot_marker, = ax_nav.plot([], [], 'bo', markersize=8)
    hist_rollout_lines = [ax_nav.plot([], [], 'gray', alpha=0.05)[0] for _ in range(4000)]
    current_rollout_lines = [ax_nav.plot([], [], 'cyan', alpha=0.4)[0] for _ in range(sim_state.ai_planner.num_rollouts)]
    best_rollout_line, = ax_nav.plot([], [], 'g-', linewidth=2, alpha=0.9)
    cost_labels = ['Total', 'Dist', 'Circ', 'Rect', 'Ctrl', 'Escape']; cost_lines = [ax_cost.plot([], [], label=l)[0] for l in cost_labels]
    ax_cost.legend(fontsize='small'); ax_nav.legend(fontsize='small')

def setup_plots():
    for ax in [ax_nav, ax_cost, ax_text]: ax.clear()
    ax_nav.set_aspect('equal'); ax_nav.set_xlim(-5, 15); ax_nav.set_ylim(-5, 15); ax_nav.grid(True); ax_nav.set_title("Interactive Simulation")
    ax_cost.grid(True); ax_cost.set_title("Live Costs"); ax_cost.set_xlabel("Step"); ax_text.axis('off')

def draw_static_elements():
    global circ_patches, rect_patches
    ax_nav.plot(sim_state.goal[0], sim_state.goal[1], 'r*', markersize=15)
    circ_patches = [plt.Circle(obs, 0.5, color='r', alpha=0.5) for obs in sim_state.circ_obstacles]; [ax_nav.add_patch(p) for p in circ_patches]
    rect_patches = [Rectangle((r['center'][0]-r['size'][0]/2,r['center'][1]-r['size'][1]/2), r['size'][0],r['size'][1],color='purple',alpha=0.5) for r in sim_state.rect_obstacles]; [ax_nav.add_patch(p) for p in rect_patches]

class DragManager:
    def __init__(self): self.dragged_artist, self.dragged_index = None, -1
drag_manager = DragManager()

def on_press(event):
    if event.inaxes!=ax_nav:return
    if sim_state.placement_mode=='rect': w,h=2.,1.; r={'center':np.array([event.xdata,event.ydata]),'size':np.array([w,h])}; sim_state.rect_obstacles.append(r); p=Rectangle((r['center'][0]-w/2,r['center'][1]-h/2),w,h,color='purple',alpha=0.5); rect_patches.append(p); ax_nav.add_patch(p); sim_state.placement_mode=None; print("Rectangle placed."); fig.canvas.draw_idle(); return
    for i,p in enumerate(circ_patches):
        if p.contains(event)[0]: drag_manager.dragged_artist,drag_manager.dragged_index=p,i; return
    obs=np.array([event.xdata,event.ydata]); sim_state.circ_obstacles.append(obs); p=plt.Circle(obs,0.5,color='r',alpha=0.5); circ_patches.append(p); ax_nav.add_patch(p); fig.canvas.draw_idle()
def on_motion(event):
    if drag_manager.dragged_artist and event.inaxes==ax_nav: pos=np.array([event.xdata,event.ydata]); drag_manager.dragged_artist.center=pos; sim_state.circ_obstacles[drag_manager.dragged_index]=pos; fig.canvas.draw_idle()
def on_release(event): drag_manager.dragged_artist=None
def on_key_press(event):
    if event.key=='r': print("Rectangle mode. Click to place."); sim_state.placement_mode='rect'
def toggle_pause(event): sim_state.is_paused=not sim_state.is_paused; pause_button.label.set_text('Resume' if sim_state.is_paused else 'Pause'); fig.canvas.draw_idle()
def stop_sim(event): sim_state.status=Status.FAILURE; plt.close(fig)
def reset_sim(event): sim_state.reset(); setup_plots(); setup_animated_artists(); draw_static_elements(); fig.canvas.draw_idle()
def save_sim(event):
    i=1;
    while os.path.exists(f"../graphs/sim_{i}"): i+=1
    save_dir=f"../graphs/sim_{i}"; plotting_utils.save_simulation_graphs(save_dir,list(sim_state.path_history),sim_state.cost_history,sim_state.goal,sim_state.circ_obstacles,sim_state.rect_obstacles)

def update(frame):
    if sim_state.status != Status.RUNNING: ani.event_source.stop(); return []
    if sim_state.is_paused:
        if sim_state.status == Status.SUCCESS and not sim_state.final_rollouts_drawn:
            for line in current_rollout_lines: line.set_data([],[]); best_rollout_line.set_data([],[])
            rollouts=sim_state.ai_planner.step(sim_state.blackboard.get("state"))
            for i,r in enumerate(rollouts):
                if i < len(current_rollout_lines):
                    end_pos=sim_state.blackboard.get("state")+r*0.5; current_rollout_lines[i].set_data([sim_state.blackboard.get("state")[0],end_pos[0]],[sim_state.blackboard.get("state")[1],end_pos[1]])
            sim_state.final_rollouts_drawn=True; fig.canvas.draw_idle()
        return []

    current_state=sim_state.blackboard.get("state"); sim_state.recent_positions.append(current_state)
    if len(sim_state.recent_positions)==sim_state.recent_positions.maxlen:
        movement=np.linalg.norm(sim_state.recent_positions[0]-sim_state.recent_positions[-1])
        if movement<0.1: sim_state.stuck_counter+=1
        else: sim_state.stuck_counter=0
    if sim_state.stuck_counter>5: sim_state.is_stuck=True
    if sim_state.is_stuck and movement>0.5: sim_state.is_stuck,sim_state.stuck_counter=False,0
    
    active_costs=sim_state.normal_cost_functions.copy()
    if sim_state.is_stuck: active_costs.update({'escape':(lambda s,a,g:escape_local_minimum_cost(s,a,g,list(sim_state.recent_positions)),10.),'distance':(distance_to_goal_cost,0.1)})
    cost_planner=CostPlanner(cost_functions=[f for f,w in active_costs.values()])
    
    rollouts=sim_state.ai_planner.step(current_state); sim_state.full_rollout_history.extend([{'s':current_state,'a':r} for r in rollouts])
    costs=[cost_planner.calculate_total_cost(current_state+act*0.5,act,sim_state.goal) for act in rollouts]; best_action=rollouts[np.argmin(costs)]
    new_state=current_state+best_action*0.5; sim_state.blackboard.set("state",new_state); sim_state.path_history.append(new_state.copy()); sim_state.ai_planner.update_beliefs(new_state)
    
    path_data=np.array(list(sim_state.path_history)); robot_path.set_data(path_data[:,0],path_data[:,1]); robot_marker.set_data([new_state[0]],[new_state[1]])
    
    hist_rollout_idx = 0
    for item in sim_state.full_rollout_history:
        if hist_rollout_idx >= len(hist_rollout_lines): break
        start,action=item['s'],item['a']; end=start+action*0.5
        hist_rollout_lines[hist_rollout_idx].set_data([start[0],end[0]],[start[1],end[1]]); hist_rollout_idx+=1
    
    for i,r in enumerate(rollouts):
        if i < len(current_rollout_lines):
            end_pos=current_state+r*0.5; current_rollout_lines[i].set_data([current_state[0],end_pos[0]],[current_state[1],end_pos[1]])
    best_rollout_line.set_data([current_state[0],(current_state+best_action*0.5)[0]],[current_state[1],(current_state+best_action*0.5)[1]])
    
    plotting_utils.plot_costs(ax_cost, sim_state.cost_history)

    status_text="Stuck!" if sim_state.is_stuck else "Running"; ax_text.clear(); ax_text.axis('off')
    ax_text.text(0.05,0.95,f"Step:{frame}|Status:{status_text}\nState:{np.round(new_state,2)}",transform=ax_text.transAxes,va='top')
    if np.linalg.norm(new_state-sim_state.goal)<0.5: sim_state.status,sim_state.is_paused=Status.SUCCESS,True; pause_button.label.set_text('Done')
    
    return [robot_path, robot_marker, best_rollout_line] + hist_rollout_lines + current_rollout_lines + cost_lines

setup_plots(); setup_animated_artists(); draw_static_elements()
fig.canvas.mpl_connect('key_press_event',on_key_press); fig.canvas.mpl_connect('button_press_event',on_press)
fig.canvas.mpl_connect('motion_notify_event',on_motion); fig.canvas.mpl_connect('button_release_event',on_release)
ax_pause=plt.axes([0.45,0.05,0.1,0.075]); pause_button=Button(ax_pause,'Start'); pause_button.on_clicked(toggle_pause)
ax_stop=plt.axes([0.56,0.05,0.1,0.075]); stop_button=Button(ax_stop,'Stop'); stop_button.on_clicked(stop_sim)
ax_reset=plt.axes([0.67,0.05,0.1,0.075]); reset_button=Button(ax_reset,'Reset'); reset_button.on_clicked(reset_sim)
ax_save=plt.axes([0.78,0.05,0.1,0.075]); save_button=Button(ax_save,'Save Sim'); save_button.on_clicked(save_sim)
ani=FuncAnimation(fig,update,frames=1000,blit=False,repeat=False,interval=50)
plt.show()