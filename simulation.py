import numpy as np
from core import Drone, GroundAsset, DroneType, DroneState

class Simulation:
    def __init__(self, scenario_config):
        self.scenario_config = scenario_config
        self.communication_enabled = scenario_config.get('communication_enabled', True)
        self.friendly_drones = []
        self.hostile_drones = []
        self.ground_assets = []
        self.all_drones = [] 
        self.sim_time = 0
        self.delta_time = 0.1 
        self.is_running = True
        
        self.setup_scenario()

    def setup_scenario(self):
        # 1. Asset (Fixed Center)
        asset_pos = np.array([0, 0, 0], dtype=float)
        self.ground_assets.append(GroundAsset("A1", asset_pos, self.scenario_config['asset_patrol_radius']))

        # 2. Friendlies (Circle formation with RANDOM Rotation)
        rotation_offset = np.random.uniform(0, 2 * np.pi)
        
        for i in range(self.scenario_config['num_friendlies']):
            angle = (i / self.scenario_config['num_friendlies']) * 2 * np.pi + rotation_offset
            radius = np.random.uniform(480, 520)
            
            pos = np.array([np.cos(angle)*radius, np.sin(angle)*radius, 100], dtype=float)
            # Pass communication_enabled from scenario config
            d = Drone(f"F{i}", DroneType.FRIENDLY, pos, self.scenario_config['communication_enabled'])
            self.friendly_drones.append(d)
            self.all_drones.append(d)

        # 3. Hostiles (Spawn Far Away with RANDOM CLUSTERING)
        for i in range(self.scenario_config['num_hostiles_ground']):
            start_x = 3000 + np.random.uniform(-500, 500)
            start_y = 3000 + np.random.uniform(-500, 500)
            pos = np.array([start_x, start_y, 100], dtype=float)
            
            d = Drone(f"HG{i}", DroneType.HOSTILE_GROUND, pos)
            self.hostile_drones.append(d)
            self.all_drones.append(d)
            
        for i in range(self.scenario_config['num_hostiles_air']):
            start_x = 3000 + np.random.uniform(-800, 800)
            start_y = 3000 + np.random.uniform(-800, 800)
            pos = np.array([start_x, start_y, 100], dtype=float)
            
            d = Drone(f"HA{i}", DroneType.HOSTILE_AIR, pos)
            self.hostile_drones.append(d)
            self.all_drones.append(d)

    def run_step(self):
        if not self.is_running: return

        # 1. Update Sensors
        for d in self.all_drones:
            d.scan(self.all_drones, self.ground_assets, self.sim_time, self.delta_time)

        # 2. Create Heartbeats BEFORE AI logic
        friendly_heartbeats = []
        if self.communication_enabled:
            for d in self.friendly_drones:
                status_code = d.get_protocol_status()
                status_str = str(status_code)
                if status_code != 0:
                    status_str = f"{status_code}:{d.last_threat_score:.1f}"
                
                friendly_heartbeats.append({
                    'id': d.id,
                    'pos': d.position.copy(),
                    'state': d.state,
                    'status': status_str
                })

        # 3. AI Logic with heartbeats (includes coordinate call checking internally)
        for d in self.all_drones:
            d.update_ai(self.sim_time, friendly_heartbeats)

        # 4. Movement, Firing & Logging
        for d in self.all_drones:
            d.update_position(self.delta_time)
            d.try_firing(self.sim_time)
            d.log_current_position(self.sim_time)

        self.sim_time += self.delta_time
        
        # 6. Check End Conditions
        friendlies_alive = any(d.state != DroneState.DESTROYED for d in self.friendly_drones)
        hostiles_alive = any(d.state != DroneState.DESTROYED for d in self.hostile_drones)
        asset_alive = self.ground_assets[0].health > 0
        
        if not friendlies_alive or not hostiles_alive or not asset_alive or self.sim_time > 6000:
            self.is_running = False

    def get_results(self):
        destroyed_f = sum(1 for d in self.friendly_drones if d.state == DroneState.DESTROYED)
        destroyed_h = sum(1 for d in self.hostile_drones if d.state == DroneState.DESTROYED)
        asset_hp = self.ground_assets[0].health
        asset_dmg = ((1000 - asset_hp) / 1000) * 100
        
        success = (asset_hp > 0) and (destroyed_h == len(self.hostile_drones))
        
        return {
            "Scenario": self.scenario_config['name'],
            "Time_Taken": f"{self.sim_time:.1f}s",
            "Mission_Success": success,
            "Asset_Damage_Avg": f"{asset_dmg:.1f}%",
            "Drones_Lost": destroyed_f,
            "Hostiles_Neutralized": destroyed_h,
            "Survivors_on_Field": len(self.friendly_drones) - destroyed_f,
            "Total_Ammo_Fired": sum(d.ammo_fired for d in self.friendly_drones),
            "RTB_Damage": sum(1 for d in self.friendly_drones if d.state == DroneState.RTB and d.rtb_reason == "Critical Damage"),
            "RTB_Fuel": 0,
            "RTB_Ammo": 0,
            "Path_Logs": {d.id: d.path_log for d in self.all_drones},
            "Battle_Logs": {d.id: d.log for d in self.all_drones}
        }
