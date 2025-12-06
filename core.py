import numpy as np
from enum import Enum
import math

# --- Technical Specifications ---
MAX_SPEED = 10.0
BASE_WEIGHT = 8.0
PAYLOAD_CAPACITY = 2.5
MIN_FIRING_RANGE = 30
MAX_FIRING_RANGE = 250
OPERATIONAL_ALT_MIN = 60
OPERATIONAL_ALT_MAX = 120
SENSOR_RANGE_MAX = 40000.0
SENSOR_FOV_DEGREES = 120.0
SENSOR_FOV_COS = np.cos(np.radians(SENSOR_FOV_DEGREES / 2.0))

# --- OPTIMIZATION CONSTANTS ---
MIN_FIRING_RANGE_SQ = MIN_FIRING_RANGE ** 2
MAX_FIRING_RANGE_SQ = MAX_FIRING_RANGE ** 2
SENSOR_RANGE_MAX_SQ = SENSOR_RANGE_MAX ** 2
THREAT_RANGE_AIR = 100.0
THREAT_RANGE_AIR_SQ = THREAT_RANGE_AIR ** 2
IDEAL_FIRING_RANGE = 120.0

# --- STATE TRANSITION THRESHOLDS ---
ASSESS_THREAT_RANGE = 1500.0  # Enemy detection range for ASSESS_THREAT
ASSESS_THREAT_RANGE_SQ = ASSESS_THREAT_RANGE ** 2

WITHDRAW_THRESHOLD = 1200.0  # Threat score for WITHDRAW
COORDINATE_ALLY_THRESHOLD = 2  # Min allies needed before transitioning from COORDINATE
DEFEND_ASSET_RANGE = 800.0  # Range to asset for DEFEND_ASSET
DEFEND_ASSET_RANGE_SQ = DEFEND_ASSET_RANGE ** 2

# --- FLOCKING & EVASION CONSTANTS ---
SEPARATION_WEIGHT = 50.0
ALIGNMENT_WEIGHT = 1.5
COHESION_WEIGHT = 0.8
AVOID_RADIUS = 15.0
EVASION_STRENGTH = 15.0 

# --- SWARM INTELLIGENCE CONSTANTS ---
ACTION_THRESHOLD = 2
CLOSE_FORMATION_RANGE = 1000.0
EASY_KILL_RANGE = 600.0      

# --- TOP CODER LOGIC CONSTANT ---
SATURATION_PENALTY_WEIGHT = 150.0

# --- MONTE CARLO COMBAT CONSTANTS ---
HIT_ACCURACY = 0.90
DAMAGE_VARIANCE = 0.15
AMMO_PER_SHOT_WEIGHT = 0.5   
FIXED_SHOT_DAMAGE = 50

class DroneType(Enum):
    FRIENDLY = 1
    HOSTILE_AIR = 2
    HOSTILE_GROUND = 3

class DroneState(Enum):
    PATROL = 1
    ASSESS_THREAT = 2
    WITHDRAW = 3
    COORDINATE = 4
    DEFEND_ASSET = 5
    RTB = 6
    DESTROYED = 7  # Can happen at ANY point

class GroundAsset:
    def __init__(self, id, position, patrol_radius):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.max_health = 1000.0
        self.health = 1000.0
        self.patrol_radius = patrol_radius 
        self.velocity = np.zeros(3, dtype=float)

    def take_damage(self, damage, sim_time): 
        self.health -= damage
        if self.health < 0:
            self.health = 0

class Drone:
    """Linear FSM: PATROL → ASSESS_THREAT → WITHDRAW → COORDINATE → DEFEND_ASSET → RTB
       DESTROYED can happen at any point during combat"""
    def __init__(self, id, drone_type, position, communication_enabled=False, ammo_per_shot_weight=AMMO_PER_SHOT_WEIGHT, shot_damage=FIXED_SHOT_DAMAGE, health=100, fuel=5000.0):
        
        self.id = id
        self.drone_type = drone_type
        self.position = np.array(position, dtype=float)          
        self.velocity = np.zeros(3, dtype=float)
        
        self.log = [] 
        self.path_log = [] 
        self.sensor_log = []
        
        self.max_speed = MAX_SPEED
        self.max_health = health
        self.health = health
        self.max_fuel = fuel
        self.max_ammo = int(PAYLOAD_CAPACITY / ammo_per_shot_weight)
        self.fuel = fuel
        self.ammo_count = self.max_ammo
        self.shot_damage = shot_damage
        
        self.last_threat_score = 0.0
        self.ground_enemy_count = 0
        self.air_enemy_count = 0

        self.fire_rate = 1.0 
        self.fire_cooldown = 0.0
        self.ammo_fired = 0 
        self.fire_visual_cooldown = 0.0 

        self.detected_enemies = {}
        self.detected_friendlies = {}
        self.detected_allies = {}
        self.known_assets = []
        
        self.communication_enabled = communication_enabled
        
        self.state = DroneState.PATROL
        self.target = None 
        self.rtb_reason = None
        self.patrol_target_pos = None
        
        self.withdraw_start_time = None
        self.coordinate_start_time = None

    # --- LOGGING UTILS ---
    def add_log(self, sim_time, event_type, details=""):
        status = self.get_protocol_status()
        status_str = str(status)
        if status in [1, 2, 3]:
             status_str = f"{status}:{self.last_threat_score:.1f}"
             
        self.log.append({
            "Time": round(sim_time, 2), 
            "Event": event_type, 
            "Details": details, 
            "Status_Code": status_str
        })
        
    def drain_log(self):
        logs = self.log
        self.log = []
        return logs

    # --- PROTOCOL MAPPING (NEW LOGIC) ---
    def get_protocol_status(self):
        """
        0 = PATROL 
        1 = Ground Threat Only (can call PATROL or status 1 with lower threat)
        2 = Air Threat Only (can call PATROL only)
        3 = Mixed Threat (can call PATROL or status 1 with lower threat)
        4 = DEFEND_ASSET (Coordinated Defense)
        5 = RTB (Mission End)
        6 = DESTROYED
        """
        if self.state == DroneState.PATROL:
            return 0
        elif self.state == DroneState.RTB:
            return 5
        elif self.state == DroneState.DESTROYED:
            return 6
        elif self.state == DroneState.DEFEND_ASSET:
            return 4
        elif self.state in [DroneState.ASSESS_THREAT, DroneState.WITHDRAW, DroneState.COORDINATE]:
            # Determine status based on enemy composition
            if self.air_enemy_count == 0 and self.ground_enemy_count > 0:
                return 1  # Ground only
            elif self.ground_enemy_count == 0 and self.air_enemy_count > 0:
                return 2  # Air only
            elif self.air_enemy_count > 0 and self.ground_enemy_count > 0:
                return 3  # Mixed threat
        return 0

    # --- SENSORS & SCANS ---
    def scan(self, all_drones, all_assets, sim_time, delta_time):
        self.detected_enemies = {}
        self.detected_friendlies = {}
        self.detected_allies = {}
        
        for other_drone in all_drones:
            if self.id == other_drone.id or other_drone.state == DroneState.DESTROYED: continue
            
            dist_sq = np.dot(other_drone.position - self.position, other_drone.position - self.position)
            if dist_sq > SENSOR_RANGE_MAX_SQ: continue 

            is_friendly = (self.drone_type == DroneType.FRIENDLY)
            is_other_friendly = (other_drone.drone_type == DroneType.FRIENDLY)

            if is_friendly:
                if is_other_friendly: self.detected_friendlies[other_drone.id] = other_drone
                else: self.detected_enemies[other_drone.id] = other_drone
            else: 
                if not is_other_friendly: self.detected_allies[other_drone.id] = other_drone
                else: self.detected_enemies[other_drone.id] = other_drone

        self.known_assets = all_assets

    # --- MAIN AI LOOP ---
    def update_ai(self, sim_time, heartbeats): 
        if self.state == DroneState.DESTROYED: return None 
        
        if self.fire_visual_cooldown > 0: self.fire_visual_cooldown -= 0.1
        if self.fire_cooldown > 0: self.fire_cooldown -= 0.1

        if self.drone_type == DroneType.FRIENDLY:
            self.run_friendly_fsm(sim_time, heartbeats) 
        else:
            self.run_hostile_ai(sim_time) 
        
        return self.target.id if (self.target is not None and hasattr(self.target, 'id')) else None

    # --- FRIENDLY LINEAR FSM ---
    def run_friendly_fsm(self, sim_time, heartbeats): 
        # 1. Critical Status Check (RTB or DESTROYED can happen anytime)
        self.check_combat_effectiveness(sim_time)
        
        if self.state in [DroneState.RTB, DroneState.DESTROYED]:
            if self.state == DroneState.RTB: self.run_rtb()
            return

        # 2. Check for coordinate calls and respond (ONLY if communication enabled)
        if self.communication_enabled and self.state == DroneState.PATROL:
            self.check_and_respond_to_calls(heartbeats, sim_time)

        # 3. Linear State Machine Flow
        if self.state == DroneState.PATROL:
            self.run_patrol(sim_time)
            
        elif self.state == DroneState.ASSESS_THREAT:
            self.run_assess_threat(sim_time, heartbeats)
            
        elif self.state == DroneState.WITHDRAW:
            # WITHDRAW only runs if communication is enabled
            if self.communication_enabled:
                self.run_withdraw(sim_time, heartbeats)
            else:
                # If somehow in WITHDRAW with communication off, go back to ASSESS_THREAT
                self.set_state(sim_time, DroneState.ASSESS_THREAT)
            
        elif self.state == DroneState.COORDINATE:
            # COORDINATE only runs if communication is enabled
            if self.communication_enabled:
                self.run_coordinate(sim_time, heartbeats)
            else:
                # If somehow in COORDINATE with communication off, go back to ASSESS_THREAT
                self.set_state(sim_time, DroneState.ASSESS_THREAT)
            
        elif self.state == DroneState.DEFEND_ASSET:
            self.run_defend_asset(sim_time)

    # --- STATE HANDLERS (LINEAR FLOW) ---
    
    def run_patrol(self, sim_time):
        """PATROL → ASSESS_THREAT when enemies detected"""
        
        # Check for enemies in range
        enemies_in_range = False
        for enemy in self.detected_enemies.values():
            if enemy.state == DroneState.DESTROYED:
                continue
            dist_sq = np.dot(enemy.position - self.position, enemy.position - self.position)
            if dist_sq <= ASSESS_THREAT_RANGE_SQ:
                enemies_in_range = True
                break
        
        if enemies_in_range:
            self.add_log(sim_time, "Transition", "PATROL → ASSESS_THREAT: Enemies detected")
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            return
        
        # Normal patrol behavior
        if self.known_assets:
            asset = self.known_assets[0]
            if self.patrol_target_pos is None or np.linalg.norm(self.position - self.patrol_target_pos) < 50:
                angle = np.random.rand() * 2 * np.pi
                r = np.random.uniform(200, asset.patrol_radius)
                self.patrol_target_pos = asset.position + np.array([np.cos(angle)*r, np.sin(angle)*r, 100])
            self.maneuver_to_point(self.patrol_target_pos, speed_multiplier=0.6)
        else:
            self.velocity *= 0.9

    def calculate_cost(self):
        """Calculate threat level and update enemy counts"""
        self.ground_enemy_count = 0
        self.air_enemy_count = 0
        
        for e in self.detected_enemies.values():
            if e.state == DroneState.DESTROYED: continue
            if e.drone_type == DroneType.HOSTILE_GROUND: self.ground_enemy_count += 1
            elif e.drone_type == DroneType.HOSTILE_AIR: self.air_enemy_count += 1
        
        if self.ground_enemy_count == 0 and self.air_enemy_count == 0:
            return 0.0
        
        if self.air_enemy_count == 0 and self.ground_enemy_count > 0:
            return 500.0  # Low threat - ground only
        
        if self.ground_enemy_count == 0 and self.air_enemy_count > 0:
            return 2000.0  # High threat - air
        
        if self.air_enemy_count > self.ground_enemy_count:
            return 1800.0  # Very high threat
        elif self.ground_enemy_count > self.air_enemy_count:
            return 800.0  # Medium threat
        else:
            return 1400.0  # High threat - mixed

    def run_assess_threat(self, sim_time, heartbeats):
        """ASSESS_THREAT → WITHDRAW if threat score high and outnumbered (ONLY if communication enabled)
           ASSESS_THREAT → DEFEND_ASSET if ground within 100m of asset + air combat active
           ASSESS_THREAT → PATROL if no threats"""
        
        active_enemies = [e for e in self.detected_enemies.values() if e.state != DroneState.DESTROYED]
        
        # No enemies - back to patrol
        if not active_enemies:
            self.add_log(sim_time, "Transition", "ASSESS_THREAT → PATROL: Threats cleared")
            self.set_state(sim_time, DroneState.PATROL)
            return
        
        # Calculate threat
        my_cost = self.calculate_cost()
        self.last_threat_score = my_cost
        
        # Check for DEFEND_ASSET trigger: ground enemies within 100m of ASSET + air enemies present
        if self.ground_enemy_count > 0 and self.air_enemy_count > 0 and self.known_assets:
            asset = self.known_assets[0]
            ground_threat_critical = False
            
            for e in self.detected_enemies.values():
                if e.state == DroneState.DESTROYED:
                    continue
                if e.drone_type == DroneType.HOSTILE_GROUND:
                    dist_to_asset_sq = np.dot(e.position - asset.position, e.position - asset.position)
                    if dist_to_asset_sq <= THREAT_RANGE_AIR_SQ:  # 100m from asset
                        ground_threat_critical = True
                        break
            
            if ground_threat_critical:
                self.add_log(sim_time, "Transition", "ASSESS_THREAT → DEFEND_ASSET: Ground within 100m of asset + air combat!")
                self.set_state(sim_time, DroneState.DEFEND_ASSET)
                return
        
        # Select target
        best_target = min(active_enemies, key=lambda e: np.linalg.norm(e.position - self.position))
        self.target = best_target
        
        # Check if need to withdraw (ONLY if communication enabled)
        if self.communication_enabled:
            nearby_allies = len(self.detected_friendlies)
            
            if my_cost >= WITHDRAW_THRESHOLD and nearby_allies < ACTION_THRESHOLD:
                self.withdraw_start_time = sim_time
                self.add_log(sim_time, "Transition", f"ASSESS_THREAT → WITHDRAW: High threat ({my_cost:.0f}), outnumbered")
                self.set_state(sim_time, DroneState.WITHDRAW)
                return
        
        # Stay in ASSESS_THREAT and engage
        nearby_allies = len(self.detected_friendlies)
        self.maneuver_intercept(evasion=False)
        self.add_log(sim_time, "Combat", f"Engaging in ASSESS_THREAT (Threat: {my_cost:.0f}, Allies: {nearby_allies})")

    def run_withdraw(self, sim_time, heartbeats):
        """WITHDRAW → COORDINATE after withdrawing for some time
           WITHDRAW → ASSESS_THREAT if reinforcements arrive"""
        
        # Check if reinforcements arrived
        nearby_allies = len(self.detected_friendlies)
        if nearby_allies >= ACTION_THRESHOLD:
            self.add_log(sim_time, "Transition", f"WITHDRAW → ASSESS_THREAT: Reinforcements arrived ({nearby_allies} allies)")
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            self.withdraw_start_time = None
            return
        
        # Check if threats cleared
        active_enemies = [e for e in self.detected_enemies.values() if e.state != DroneState.DESTROYED]
        if not active_enemies:
            self.add_log(sim_time, "Transition", "WITHDRAW → PATROL: Threats cleared during withdrawal")
            self.set_state(sim_time, DroneState.PATROL)
            self.withdraw_start_time = None
            return
        
        # Withdraw for 5 seconds then call for help
        if self.withdraw_start_time and (sim_time - self.withdraw_start_time) >= 5.0:
            self.coordinate_start_time = sim_time
            self.add_log(sim_time, "Transition", "WITHDRAW → COORDINATE: Requesting reinforcements")
            self.set_state(sim_time, DroneState.COORDINATE)
            self.withdraw_start_time = None
            return
        
        # Continue withdrawing
        if self.target is None:
            self.target = active_enemies[0]
        
        vec_away = self.position - self.target.position
        dist = np.linalg.norm(vec_away)
        if dist > 0: vec_away /= dist
        
        vec_safe = np.array([0,0,100]) - self.position
        d_safe = np.linalg.norm(vec_safe)
        if d_safe > 0: vec_safe /= d_safe

        final_vec = (vec_away * 0.7) + (vec_safe * 0.3)
        final_vec = (final_vec / np.linalg.norm(final_vec)) * self.max_speed
        
        self.velocity = (self.velocity * 0.8) + (final_vec * 0.2)

    def run_coordinate(self, sim_time, heartbeats):
        """COORDINATE → DEFEND_ASSET when allies arrive
           COORDINATE → PATROL if threats cleared"""
        
        # Check if threats cleared
        active_enemies = [e for e in self.detected_enemies.values() if e.state != DroneState.DESTROYED]
        if not active_enemies:
            self.add_log(sim_time, "Transition", "COORDINATE → PATROL: Threats eliminated")
            self.set_state(sim_time, DroneState.PATROL)
            self.coordinate_start_time = None
            return
        
        # Check if reinforcements arrived
        nearby_allies = len(self.detected_friendlies)
        if nearby_allies >= COORDINATE_ALLY_THRESHOLD:
            self.add_log(sim_time, "Transition", f"COORDINATE → DEFEND_ASSET: {nearby_allies} allies responded")
            if self.known_assets:
                self.target = self.known_assets[0]
            self.set_state(sim_time, DroneState.DEFEND_ASSET)
            self.coordinate_start_time = None
            return
        
        # Hold position and fight defensively
        nearest_enemy = min(active_enemies, key=lambda e: np.linalg.norm(e.position - self.position))
        self.target = nearest_enemy
        self.maneuver_intercept(evasion=True)
        
        # Broadcast every 2 seconds
        if self.coordinate_start_time and (sim_time - self.coordinate_start_time) % 2.0 < 0.1:
            self.add_log(sim_time, "Broadcast", f"COORDINATE active: Need backup! (Allies: {nearby_allies})")

    def check_and_respond_to_calls(self, heartbeats, sim_time):
        """Check for coordinate calls and respond based on status code rules"""
        if not heartbeats:
            return
        
        my_status = self.get_protocol_status()
        
        # Only PATROL drones can respond
        if my_status != 0:
            return
        
        for hb in heartbeats:
            if hb['id'] == self.id:
                continue
            
            # Check if drone is in COORDINATE state
            if hb.get('state') != DroneState.COORDINATE:
                continue
            
            # Parse caller's status
            caller_status_str = hb.get('status', '0')
            try:
                if ':' in caller_status_str:
                    caller_status = int(caller_status_str.split(':')[0])
                    caller_threat = float(caller_status_str.split(':')[1])
                else:
                    caller_status = int(caller_status_str)
                    caller_threat = 0.0
            except:
                continue
            
            # Apply response rules based on caller's status code
            should_respond = False
            
            if caller_status == 1:  # Ground threat only
                # Can call PATROL (status 0) - we are patrol, so respond
                should_respond = True
                
            elif caller_status == 2:  # Air threat only
                # Can ONLY call PATROL (status 0) - we are patrol, so respond
                should_respond = True
                
            elif caller_status == 3:  # Mixed threat
                # Can call PATROL (status 0) - we are patrol, so respond
                should_respond = True
            
            if should_respond:
                call_pos = np.array(hb['pos'], dtype=float)
                dist = np.linalg.norm(call_pos - self.position)
                self.add_log(sim_time, "Response", f"Responding to COORDINATE call (Status {caller_status}, Dist: {dist:.0f}m)")
                # Move towards the calling drone
                self.patrol_target_pos = call_pos
                return

    def run_defend_asset(self, sim_time):
        """DEFEND_ASSET → RTB when mission complete or critical
           DEFEND_ASSET → PATROL when area secured"""
        
        # Check if all threats eliminated
        active_enemies = [e for e in self.detected_enemies.values() if e.state != DroneState.DESTROYED]
        
        if not active_enemies:
            self.add_log(sim_time, "Transition", "DEFEND_ASSET → PATROL: Area secured")
            self.set_state(sim_time, DroneState.PATROL)
            return
        
        # Defend the asset
        if self.known_assets:
            asset = self.known_assets[0]
            
            # Find closest threat to asset
            nearest_threat = min(active_enemies, 
                               key=lambda e: np.linalg.norm(e.position - asset.position))
            
            self.target = nearest_threat
            
            # Position between threat and asset
            dist_to_asset = np.linalg.norm(self.position - asset.position)
            
            if dist_to_asset < DEFEND_ASSET_RANGE:
                # Close to asset - intercept threats
                self.maneuver_intercept(evasion=False)
            else:
                # Too far - move closer to asset
                self.maneuver_to_point(asset.position, speed_multiplier=1.0)
        else:
            # No asset - just engage
            self.target = active_enemies[0]
            self.maneuver_intercept(evasion=False)

    # --- PHYSICS & COMBAT ---
    def maneuver_intercept(self, evasion=False):
        if self.target is None: 
            return
        
        if not hasattr(self.target, 'velocity'):
            self.maneuver_to_point(self.target.position, 1.0, evasion)
            return

        dist = np.linalg.norm(self.target.position - self.position)
        time_to_hit = dist / (MAX_SPEED + 1e-6)
        predicted_pos = self.target.position + (self.target.velocity * time_to_hit)
        
        if self.known_assets and hasattr(self.target, 'drone_type') and self.target.drone_type == DroneType.HOSTILE_GROUND:
            asset_pos = self.known_assets[0].position
            vec_to_asset = asset_pos - self.target.position
            d_asset = np.linalg.norm(vec_to_asset)
            if d_asset > IDEAL_FIRING_RANGE:
                predicted_pos = self.target.position + (vec_to_asset / d_asset) * (d_asset * 0.6)

        vec_to_fut = predicted_pos - self.position
        d_fut = np.linalg.norm(vec_to_fut)
        target_spot = predicted_pos
        if d_fut > 0:
            target_spot = predicted_pos - (vec_to_fut / d_fut) * IDEAL_FIRING_RANGE

        self.maneuver_to_point(target_spot, 1.0, evasion)

    def maneuver_to_point(self, target_pos, speed_multiplier, evasion=False):
        desired = target_pos - self.position
        dist = np.linalg.norm(desired)
        if dist > 0: desired = (desired / dist) * (self.max_speed * speed_multiplier)
        
        steering = desired - self.velocity
        if np.linalg.norm(steering) > 5.0: steering = (steering / np.linalg.norm(steering)) * 5.0
        
        self.velocity += steering
        
        if self.position[2] < OPERATIONAL_ALT_MIN: self.velocity[2] += 2.0
        elif self.position[2] > OPERATIONAL_ALT_MAX: self.velocity[2] -= 2.0

    # --- STATUS UPDATES ---
    def update_specs(self, health=None, ammo=None, altitude=None, speed=None):
        if health is not None: self.max_health = float(health); self.health = float(health)
        if ammo is not None: self.ammo_count = int(ammo)

    def check_combat_effectiveness(self, sim_time):
        """Can trigger RTB or DESTROYED at any point"""
        if self.state in [DroneState.RTB, DroneState.DESTROYED]: 
            return
        
        if self.ammo_count <= 0:
            self.set_state(sim_time, DroneState.RTB)
            self.add_log(sim_time, "RTB", "Ammo Depleted (Mission End)")
            self.rtb_reason = "Ammo Depleted"
            
        elif self.fuel < 200 or self.health < 20:
            self.set_state(sim_time, DroneState.RTB)
            self.add_log(sim_time, "RTB", "Critical Levels (Mission End)")
            self.rtb_reason = "Critical Damage" if self.health < 20 else "Low Fuel"

    def run_rtb(self):
        self.maneuver_to_point(np.array([0,0,100]), 1.0)

    def set_state(self, sim_time, new_state):
        if self.state != new_state:
            self.add_log(sim_time, "State Change", f"{self.state.name} → {new_state.name}")
            self.state = new_state

    # --- COMBAT ---
    def fire_shot(self):
        if self.ammo_count > 0:
            self.ammo_count -= 1
            self.fire_cooldown = self.fire_rate
            self.ammo_fired += 1
            self.fire_visual_cooldown = 0.3
            return True
        return False
    
    def take_damage(self, damage, sim_time):
        """DESTROYED can happen at ANY point during combat"""
        if self.state != DroneState.DESTROYED:
            self.health -= damage
            if self.health <= 0:
                self.health = 0
                self.add_log(sim_time, "DESTROYED", f"Eliminated (was in {self.state.name})")
                self.set_state(sim_time, DroneState.DESTROYED) 
                self.velocity = np.zeros(3)

    def try_firing(self, sim_time):
        if self.target is None or isinstance(self.target, np.ndarray): 
            return
        if hasattr(self.target, 'drone_type') and self.target.drone_type == self.drone_type: 
            return
        
        if self.fire_cooldown <= 0:
            dist_sq = np.dot(self.position - self.target.position, self.position - self.target.position)
            if dist_sq >= MIN_FIRING_RANGE_SQ and dist_sq <= MAX_FIRING_RANGE_SQ:
                if self.fire_shot():
                    if np.random.random() < HIT_ACCURACY:
                        variance = np.random.uniform(1.0 - DAMAGE_VARIANCE, 1.0 + DAMAGE_VARIANCE)
                        actual_damage = self.shot_damage * variance
                        self.target.take_damage(actual_damage, sim_time)
                    else:
                        self.add_log(sim_time, "Miss", "Shot missed target")

    def update_position(self, delta_time):
        if self.state == DroneState.DESTROYED: return
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > self.max_speed: self.velocity = (self.velocity / vel_norm) * self.max_speed
        self.position += self.velocity * delta_time
        self.fuel -= vel_norm * delta_time * 0.01

    def log_current_position(self, sim_time):
        if self.state != DroneState.DESTROYED:
            status = self.get_protocol_status()
            status_str = str(status)
            if status in [1, 2, 3]:
                status_str = f"{status}:{self.last_threat_score:.1f}"
            
            self.path_log.append({
                "time": round(sim_time, 1),
                "position": list(self.position),
                "health": round(self.health, 1),
                "ammo": self.ammo_count,
                "state": self.state.name,
                "status_code": status_str 
            })
            
    def run_hostile_ai(self, sim_time):
        if self.state == DroneState.DESTROYED: return
        
        target = None
        if self.drone_type == DroneType.HOSTILE_GROUND and self.known_assets:
            target = self.known_assets[0]
        elif self.drone_type == DroneType.HOSTILE_AIR:
            if self.detected_enemies:
                target = min(self.detected_enemies.values(), key=lambda e: np.linalg.norm(e.position - self.position))
            elif self.known_assets:
                target = self.known_assets[0]

        if target:
            self.target = target
            self.maneuver_intercept(evasion=False)
            self.try_firing(sim_time)
        else:
            self.velocity *= 0.9
