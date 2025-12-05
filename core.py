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
IDEAL_FIRING_RANGE = 150.0

# --- FLOCKING & EVASION CONSTANTS ---
SEPARATION_WEIGHT = 50.0
ALIGNMENT_WEIGHT = 1.5
COHESION_WEIGHT = 0.8
AVOID_RADIUS = 15.0
EVASION_STRENGTH = 15.0 

# --- SWARM INTELLIGENCE CONSTANTS ---
ACTION_THRESHOLD = 3          # Neighbors needed to feel safe
CLOSE_FORMATION_RANGE = 800.0 # Range to count as a "Neighbor"
EASY_KILL_RANGE = 600.0      

# --- TOP CODER LOGIC CONSTANT ---
# 400.0 means: "I'd rather fly 400m further than attack a target my friend is already hitting."
SATURATION_PENALTY_WEIGHT = 400.0 

# --- MONTE CARLO COMBAT CONSTANTS (ORIGINAL HIGH ACCURACY) ---
HIT_ACCURACY = 0.90          # 90% Accuracy
DAMAGE_VARIANCE = 0.15       # +/- 15% Damage
# 2.5kg / 0.5kg = 5 Shots (Very limited ammo, requires smart usage)
AMMO_PER_SHOT_WEIGHT = 0.5   
FIXED_SHOT_DAMAGE = 50       # Base damage (2-3 hits to kill 100HP target)

class DroneType(Enum):
    FRIENDLY = 1
    HOSTILE_AIR = 2
    HOSTILE_GROUND = 3

class DroneState(Enum):
    PATROL = 1
    ASSESS_THREAT = 2
    ENGAGE = 3
    DEFEND_ASSET = 4
    RTB = 6
    DESTROYED = 7
    ATTACKING = 8
    # REARM Removed - One way mission only
    WITHDRAW = 10 

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
    """Implements ISP with Priority Cost Hierarchy & Smart Reinforcement"""
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
        
        # Stores the last calculated threat score for logging & broadcasting
        self.last_threat_score = 0.0

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

    # --- LOGGING UTILS ---
    def add_log(self, sim_time, event_type, details=""):
        status = self.get_protocol_status()
        status_str = str(status)
        if status != 0:
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

    # --- PROTOCOL MAPPING ---
    def get_protocol_status(self):
        """
        0 = PATROL 
        1 = ENGAGE (Withdraw = False)
        2 = PINNED/WITHDRAW/RTB (Withdraw = True)
        """
        if self.state in [DroneState.PATROL, DroneState.ASSESS_THREAT]:
            return 0
        elif self.state in [DroneState.ENGAGE, DroneState.DEFEND_ASSET, DroneState.ATTACKING]:
            return 1
        elif self.state in [DroneState.WITHDRAW, DroneState.RTB, DroneState.DESTROYED]:
            return 2
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
        
        return self.target.id if (self.target and hasattr(self.target, 'id')) else None

    # --- FRIENDLY FSM ---
    def run_friendly_fsm(self, sim_time, heartbeats): 
        # 1. Critical Status Check (One-way RTB)
        self.check_combat_effectiveness(sim_time)
        
        # If RTB or Dead, we stop all logic and just exit/fly home.
        if self.state in [DroneState.RTB, DroneState.DESTROYED]:
            if self.state == DroneState.RTB: self.run_rtb()
            return

        # 2. Check for Reinforcement Calls (ORIGINAL NAME: check_reinforcements)
        if self.state in [DroneState.PATROL, DroneState.ASSESS_THREAT, DroneState.ENGAGE, DroneState.WITHDRAW]:
            call_target = self.check_reinforcements(heartbeats, sim_time)
            if call_target:
                self.target = call_target
                self.set_state(sim_time, DroneState.DEFEND_ASSET) 
                return

        # 3. Standard State Machine
        if self.state == DroneState.PATROL:
            self.run_patrol(sim_time)
        elif self.state == DroneState.ASSESS_THREAT:
            self.run_assess_threat(sim_time, heartbeats)
        elif self.state == DroneState.ENGAGE:
            self.run_engage(sim_time)
        elif self.state == DroneState.DEFEND_ASSET:
            self.run_defend_asset(sim_time)
        elif self.state == DroneState.WITHDRAW:
            self.run_withdraw(sim_time, heartbeats)

    # --- REINFORCEMENT LOGIC (ORIGINAL NAME) ---
    def check_reinforcements(self, heartbeats, sim_time):
        """
        Determines if this drone should answer an 'Engage Call'.
        """
        my_status = self.get_protocol_status()
        my_score = self.last_threat_score
        
        if my_status == 2 and self.state != DroneState.WITHDRAW: return None

        best_call_pos = None
        highest_urgency = -1.0

        for hb in heartbeats:
            caller_id = hb['id']
            if caller_id == self.id: continue
            
            hb_code = str(hb.get('status', '0'))
            if ':' not in hb_code: continue 
            
            try:
                s_code, s_score = hb_code.split(':')
                caller_status = int(s_code)
                caller_score = float(s_score)
            except:
                continue

            should_help = False
            
            if my_status == 0:
                should_help = True
            elif my_status == 1:
                if caller_score > my_score:
                    should_help = True
            
            if should_help and caller_score > highest_urgency:
                highest_urgency = caller_score
                best_call_pos = Drone(caller_id, DroneType.FRIENDLY, hb['pos'])

        return best_call_pos

    # --- STATE HANDLERS ---
    def run_patrol(self, sim_time):
        if any(e.state != DroneState.DESTROYED for e in self.detected_enemies.values()):
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            return

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
        ground_enemies = 0
        air_enemies = 0
        
        for e in self.detected_enemies.values():
            if e.state == DroneState.DESTROYED: continue
            if e.drone_type == DroneType.HOSTILE_GROUND: ground_enemies += 1
            elif e.drone_type == DroneType.HOSTILE_AIR: air_enemies += 1
        
        if ground_enemies == 0 and air_enemies == 0: return 0.0
        if air_enemies == 0 and ground_enemies > 0: return 0.0
        if ground_enemies > air_enemies: return 600.0
        if air_enemies > ground_enemies: return 1400.0
        if ground_enemies == 0 and air_enemies > 0: return 2000.0
            
        return 1000.0 

    def get_saturation_penalty(self, target_id):
        """
        Calculates how 'crowded' a target is.
        """
        attackers = 0
        for friend in self.detected_friendlies.values():
            if friend.state in [DroneState.ENGAGE, DroneState.ATTACKING]:
                if friend.target and hasattr(friend.target, 'id') and friend.target.id == target_id:
                    attackers += 1
        return attackers * SATURATION_PENALTY_WEIGHT

    def run_assess_threat(self, sim_time, heartbeats):
        active_enemies = [e for e in self.detected_enemies.values() if e.state != DroneState.DESTROYED]
        if not active_enemies:
            self.set_state(sim_time, DroneState.PATROL)
            return

        # --- RESERVE GUARD LOGIC (NEW) ---
        # Strategy: Keep 2 drones back as reserves until our numbers drop to 4 (50%).
        all_ids = [self.id] + [hb['id'] for hb in heartbeats]
        all_ids.sort() # Deterministic ordering (F0, F1, F2...)
        
        survivor_count = len(all_ids)
        
        # If we have healthy numbers (> 4), the last 2 drones in the list hold back.
        if survivor_count > 4:
            my_rank = all_ids.index(self.id)
            if my_rank >= (survivor_count - 2):
                # I am a reserve unit!
                # Instead of engaging, hold position near the base/asset.
                self.add_log(sim_time, "Reserve", "Holding Base Position (Reserve Guard)")
                # Maintain altitude, hover near 0,0
                self.maneuver_to_point(np.array([0, 0, 100]), 1.0)
                return 
        # ---------------------------------

        # END GAME LOGIC: If only 1 enemy, Kill. If many, Coordinate.
        is_end_game = (len(active_enemies) == 1)

        best_target = None
        best_score = float('inf')

        for enemy in active_enemies:
            dist = np.linalg.norm(enemy.position - self.position)
            type_bias = -10000.0 if enemy.drone_type == DroneType.HOSTILE_GROUND else 0.0
            
            # Smart Saturation
            if is_end_game:
                penalty = 0.0 # KILL MODE: No hesitation
            else:
                penalty = self.get_saturation_penalty(enemy.id) # SWARM MODE: Spread out
            
            final_score = dist + type_bias + penalty
            
            if final_score < best_score:
                best_score = final_score
                best_target = enemy

        self.target = best_target
        
        # [PRESERVED PRIORITY LOGIC]
        my_cost = self.calculate_cost()
        self.last_threat_score = my_cost
        
        if my_cost >= 2000.0:
            nearby_allies = 0
            for hb in heartbeats:
                if np.linalg.norm(np.array(hb['pos']) - self.position) < CLOSE_FORMATION_RANGE:
                    nearby_allies += 1
            
            if nearby_allies < ACTION_THRESHOLD:
                self.set_state(sim_time, DroneState.WITHDRAW)
                self.add_log(sim_time, "Distress", f"Pinned by Air. Cost: {my_cost}")
            else:
                self.set_state(sim_time, DroneState.ENGAGE)
        else:
            self.set_state(sim_time, DroneState.ENGAGE)
            if my_cost == 0.0:
                self.add_log(sim_time, "Engage", "Priority: Ground Only")
            else:
                self.add_log(sim_time, "Engage", f"Priority: Mixed (Cost {my_cost})")

    def run_engage(self, sim_time):
        if not self.target or self.target.state == DroneState.DESTROYED:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            return
        self.maneuver_intercept(evasion=False)

    def run_defend_asset(self, sim_time):
        if not self.target:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            return
            
        self.maneuver_to_point(self.target.position, speed_multiplier=1.0, evasion=True)
        
        if np.linalg.norm(self.position - self.target.position) < 200:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)

    def run_withdraw(self, sim_time, heartbeats):
        nearby_allies = 0
        for hb in heartbeats:
            if np.linalg.norm(np.array(hb['pos']) - self.position) < CLOSE_FORMATION_RANGE:
                nearby_allies += 1
                
        if nearby_allies >= ACTION_THRESHOLD:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            self.add_log(sim_time, "Re-Engage", "Reinforcements Arrived")
            return

        if not self.target:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)
            return

        vec_away = self.position - self.target.position
        dist = np.linalg.norm(vec_away)
        if dist > 0: vec_away /= dist
        
        vec_safe = np.array([0,0,100]) - self.position
        d_safe = np.linalg.norm(vec_safe)
        if d_safe > 0: vec_safe /= d_safe

        final_vec = (vec_away * 0.6) + (vec_safe * 0.4)
        final_vec = (final_vec / np.linalg.norm(final_vec)) * self.max_speed
        
        self.velocity = (self.velocity * 0.8) + (final_vec * 0.2)
        
        if dist > 800:
            self.set_state(sim_time, DroneState.ASSESS_THREAT)

    # --- PHYSICS & COMBAT (AGGRESSIVE STRAIGHT LINE) ---
    def maneuver_intercept(self, evasion=False):
        if not self.target: return
        
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
                predicted_pos = self.target.position + (vec_to_asset / d_asset) * (d_asset * 0.7)

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

    # --- UPDATED: ONE-WAY RTB LOGIC (No Rearm Loop) ---
    def check_combat_effectiveness(self, sim_time):
        if self.state in [DroneState.RTB, DroneState.DESTROYED]: return
        
        # If ammo is out -> RTB (Mission Over)
        if self.ammo_count <= 0:
            self.set_state(sim_time, DroneState.RTB)
            self.add_log(sim_time, "RTB", "Ammo Depleted (Mission End)")
            self.rtb_reason = "Ammo Depleted"
            
        # If Fuel/Health Critical -> RTB (Mission Over)
        elif self.fuel < 200 or self.health < 30:
            self.set_state(sim_time, DroneState.RTB)
            self.add_log(sim_time, "RTB", "Critical Levels (Mission End)")
            self.rtb_reason = "Critical Damage" if self.health < 30 else "Low Fuel"

    def run_rtb(self):
        # Fly to base and stay there.
        self.maneuver_to_point(np.array([0,0,100]), 1.0)

    def set_state(self, sim_time, new_state):
        if self.state != new_state:
            self.add_log(sim_time, "State Change", f"{self.state.name} -> {new_state.name}")
            self.state = new_state

    # --- ORIGINAL COMBAT (HIGH ACCURACY / VARIANCE) ---
    def fire_shot(self):
        if self.ammo_count > 0:
            self.ammo_count -= 1
            self.fire_cooldown = self.fire_rate
            self.ammo_fired += 1
            self.fire_visual_cooldown = 0.3
            return True
        return False
    
    def take_damage(self, damage, sim_time):
        if self.state != DroneState.DESTROYED:
            self.health -= damage
            if self.health <= 0:
                self.health = 0
                self.set_state(sim_time, DroneState.DESTROYED) 
                self.velocity = np.zeros(3)

    def try_firing(self, sim_time):
        if not self.target: return
        if hasattr(self.target, 'drone_type') and self.target.drone_type == self.drone_type: return
        
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
            if status != 0:
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