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

        self.sim_time = 0.0
        self.delta_time = 0.1
        self.is_running = True

        self.setup_scenario()

    def setup_scenario(self):
        # 1. Ground Asset at origin
        asset_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.ground_assets.append(
            GroundAsset("A1", asset_pos, self.scenario_config["asset_patrol_radius"])
        )

        # 2. Friendlies – circular formation around asset with random rotation & radius jitter
        rotation_offset = np.random.uniform(0, 2 * np.pi)

        num_friend = self.scenario_config["num_friendlies"]
        for i in range(num_friend):
            angle = (i / num_friend) * 2 * np.pi + rotation_offset
            radius = np.random.uniform(480.0, 520.0)
            pos = np.array(
                [np.cos(angle) * radius, np.sin(angle) * radius, 100.0],
                dtype=float,
            )
            d = Drone(f"F{i}", DroneType.FRIENDLY, pos, self.communication_enabled)
            self.friendly_drones.append(d)
            self.all_drones.append(d)

        # 3. Hostile ground attackers – spawn in a tighter cluster, focused on asset
        for i in range(self.scenario_config["num_hostiles_ground"]):
            start_x = 3000.0 + np.random.uniform(-400.0, 400.0)
            start_y = 3000.0 + np.random.uniform(-400.0, 400.0)
            pos = np.array([start_x, start_y, 80.0], dtype=float)
            d = Drone(f"HG{i}", DroneType.HOSTILE_GROUND, pos)
            self.hostile_drones.append(d)
            self.all_drones.append(d)

        # 4. Hostile air threats – slightly wider dispersion
        for i in range(self.scenario_config["num_hostiles_air"]):
            start_x = 3200.0 + np.random.uniform(-800.0, 800.0)
            start_y = 3200.0 + np.random.uniform(-800.0, 800.0)
            pos = np.array([start_x, start_y, 150.0], dtype=float)
            d = Drone(f"HA{i}", DroneType.HOSTILE_AIR, pos)
            self.hostile_drones.append(d)
            self.all_drones.append(d)

    # ---------------------------------------------------
    # HEARTBEAT LOGIC – THIS UNBLOCKS YOUR C-COST LOGIC
    # ---------------------------------------------------
    def build_friendly_heartbeats(self):
        """
        Heartbeat format must match what core.check_reinforcements() expects:
        hb['status'] is a string: "code" or "code:score"
        where:
            code = 0 (PATROL), 1 (ENGAGE), 2 (WITHDRAW/RTB)
            score = last_threat_score (your ccost)
        """
        if not self.communication_enabled:
            return []

        heartbeats = []
        for d in self.friendly_drones:
            status_code = d.get_protocol_status()
            # Default: 0 -> just "0"
            if status_code == 0:
                status_str = "0"
            else:
                # e.g., "1:1400.0" or "2:2000.0"
                status_str = f"{status_code}:{d.last_threat_score:.1f}"

            heartbeats.append(
                {
                    "id": d.id,
                    "pos": d.position.copy(),
                    "state": d.state,
                    "status": status_str,
                }
            )
        return heartbeats

    def run_step(self):
        if not self.is_running:
            return

        # 1. Sensor update for ALL drones
        for d in self.all_drones:
            d.scan(self.all_drones, self.ground_assets, self.sim_time, self.delta_time)

        # 2. Build shared friendly heartbeats (for engage_call / WITHDRAW logic)
        friendly_heartbeats = self.build_friendly_heartbeats()

        # 3. AI decision step (friendlies use heartbeats, hostiles ignore them)
        for d in self.all_drones:
            d.update_ai(self.sim_time, friendly_heartbeats)

        # 4. Physics + Firing + Logging
        for d in self.all_drones:
            d.update_position(self.delta_time)

            # Friendlies fire here (their AI picks target, this executes weapons)
            # Hostiles already fire from run_hostile_ai() inside core.py
            if d.drone_type == DroneType.FRIENDLY:
                d.try_firing(self.sim_time)

            d.log_current_position(self.sim_time)

        # 5. Advance time
        self.sim_time += self.delta_time

        # 6. Check termination conditions
        self.check_end_conditions()

    def check_end_conditions(self):
        friendlies_alive = any(
            d.state != DroneState.DESTROYED for d in self.friendly_drones
        )
        hostiles_alive = any(
            d.state != DroneState.DESTROYED for d in self.hostile_drones
        )
        asset_alive = self.ground_assets[0].health > 0

        # Hard time cap
        if (
            not friendlies_alive
            or not hostiles_alive
            or not asset_alive
            or self.sim_time > 600.0
        ):
            self.is_running = False

    def get_results(self):
        destroyed_f = sum(
            1 for d in self.friendly_drones if d.state == DroneState.DESTROYED
        )
        destroyed_h = sum(
            1 for d in self.hostile_drones if d.state == DroneState.DESTROYED
        )

        asset_hp = self.ground_assets[0].health
        asset_dmg = ((1000.0 - asset_hp) / 1000.0) * 100.0

        success = (asset_hp > 0) and (destroyed_h == len(self.hostile_drones))

        return {
            "Scenario": self.scenario_config["name"],
            "Time_Taken": f"{self.sim_time:.1f}s",
            "Mission_Success": success,
            "Asset_Damage_Avg": f"{asset_dmg:.1f}%",
            "Drones_Lost": destroyed_f,
            "Hostiles_Neutralized": destroyed_h,
            "Survivors_on_Field": len(self.friendly_drones) - destroyed_f,
            "Total_Ammo_Fired": sum(d.ammo_fired for d in self.friendly_drones),
            # RTB breakdowns – your core sets rtb_reason
            "RTB_Damage": sum(
                1
                for d in self.friendly_drones
                if d.state == DroneState.RTB and d.rtb_reason == "Critical Damage"
            ),
            "RTB_Fuel": sum(
                1
                for d in self.friendly_drones
                if d.state == DroneState.RTB and d.rtb_reason == "Low Fuel"
            ),
            "RTB_Ammo": sum(
                1
                for d in self.friendly_drones
                if d.state == DroneState.RTB and d.rtb_reason == "Ammo Depleted"
            ),
            "Path_Logs": {d.id: d.path_log for d in self.all_drones},
            "Battle_Logs": {d.id: d.log for d in self.all_drones},
        }
