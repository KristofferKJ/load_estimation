# ========== ROS2 ==========
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


import socket
import time
import math


# ========== UDP Communication Setup ==========
UDP_IP = "192.168.144.25"
UDP_PORT = 37260
LOCAL_PORT = 37261  # Local port to receive responses

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", LOCAL_PORT))  # Bind to local port to receive data
sock.settimeout(0.05)  # Timeout for receiving data

def send(cmd):
    if isinstance(cmd, str):
        sock.sendto(bytes.fromhex(cmd), (UDP_IP, UDP_PORT))
    else:
        sock.sendto(bytes(cmd), (UDP_IP, UDP_PORT))

def decodeMsg(msg):
    data = None
        
    if not isinstance(msg, str):
        print("Message is not string")
        return data

    # 10 bytes: STX+CTRL+Data_len+SEQ+CMD_ID+CRC16
    #            2 + 1  +    2   + 2 +   1  + 2
    MINIMUM_DATA_LENGTH=10*2
    if len(msg)<MINIMUM_DATA_LENGTH:
        print("Message too short")
        return data
    
    # Now we got minimum amount of data. Check if we have enough
    # Data length, bytes are reversed, according to SIYI SDK
    low_b = msg[6:8] # low byte
    high_b = msg[8:10] # high byte
    data_len = high_b+low_b
    data_len = int('0x'+data_len, base=16)
    char_len = data_len*2 # number of characters. Each byte is represented by two characters in hex, e.g. '0A'= 2 chars

    # check crc16, if msg is OK!
    msg_crc=msg[-4:] # last 4 characters
    payload=msg[:-4]
    payload = bytes.fromhex(payload)
    expected_crc=crc16_xmodem(payload)
    if expected_crc!=int(msg_crc, 16).to_bytes(2, 'big'):
        print(f"CRC16 mismatch: expected {expected_crc}, got {int(msg_crc, 16).to_bytes(2, 'big')}")
        return data
    
    # Sequence
    low_b = msg[10:12] # low byte
    high_b = msg[12:14] # high byte
    seq_hex = high_b+low_b
    seq = int('0x'+seq_hex, base=16)
    
    # CMD ID
    cmd_id = msg[14:16]

    # DATA
    if data_len>0:
        data = msg[16:16+char_len]
    else:
        data=''

    return data, data_len, cmd_id, seq

def receive():
    try:
        buff, addr = sock.recvfrom(1024)  # Buffer size
        buff_str = buff.hex()

        MINIMUM_DATA_LENGTH=10*2

        HEADER='5566'
        # Go through the buffer
        while(len(buff_str)>=MINIMUM_DATA_LENGTH):
            if buff_str[0:4]!=HEADER:
                # Remove the 1st element and continue 
                tmp=buff_str[1:]
                buff_str=tmp
                continue

            # Now we got minimum amount of data. Check if we have enough
            # Data length, bytes are reversed, according to SIYI SDK
            low_b = buff_str[6:8] # low byte
            high_b = buff_str[8:10] # high byte
            data_len = high_b+low_b
            data_len = int('0x'+data_len, base=16)
            char_len = data_len*2

            # Check if there is enough data (including payload)
            if(len(buff_str) < (MINIMUM_DATA_LENGTH+char_len)):
                # No useful data
                buff_str=''
                break

            packet = buff_str[0:MINIMUM_DATA_LENGTH+char_len]
            buff_str = buff_str[MINIMUM_DATA_LENGTH+char_len:]

            # Finally decode the packet!
            val = decodeMsg(packet)

            if val is None:
                continue

            data, data_len, cmd_id, seq = val[0], val[1], val[2], val[3]
            return data, data_len, cmd_id, seq
        return
    except socket.timeout:
        print("No response received")
        return None
    
def crc16_xmodem(data: bytes, poly=0x1021, init=0x0000):
        crc = init
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc.to_bytes(2, 'little')  # low byte first

def toInt(hexval):
        """
        Converts hexidecimal value to an integer number, which can be negative
        Ref: https://www.delftstack.com/howto/python/python-hex-to-int/

        Params
        --
        hexval: [string] String of the hex value
        """
        bits = 16
        val = int(hexval, bits)
        if val & (1 << (bits-1)):
            val -= 1 << bits
        return val

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.last_error = 0
        self.last_time = None

    def update(self, measurement):
        current_time = time.time()
        error = self.setpoint - measurement

        delta_time = current_time - self.last_time if self.last_time else 0
        delta_error = error - self.last_error if self.last_time else 0

        if delta_time > 0:
            self.integral += error * delta_time
            derivative = delta_error / delta_time
        else:
            derivative = 0

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.last_error = error
        self.last_time = current_time

        return output


class GimbalControlNode(Node):
    def __init__(self):
        super().__init__('gimbal_control_node')
        self.get_logger().info("Gimbal Control Node has been started.")

        self.PID_yaw = (10, 0, 1.0)   # P, I, D values for yaw
        self.PID_pitch = (10, 0, 1.0) # P, I, D values for pitch
        self.pid_yaw = PIDController(*self.PID_yaw)
        self.pid_pitch = PIDController(*self.PID_pitch)

        self.center_gimbal()
        time.sleep(2) # wait for gimbal to center

        self.setpoint_1 = [0, 180] # yaw, pitch
        self.setpoint_2 = [90, 135] # yaw, pitch
        self.setpoint = self.setpoint_1

        

        # ========== Publishers ==========
        self.publish_load_angles_ = self.create_publisher(Vector3, '/gimbal/load_angles', 2)

        # ========== Subscribers ==========
        self.subscribe_setpoint_ = self.create_subscription(Vector3, '/gimbal/error', self.setpoint_callback, 2)

        # ========== timers ==========
        timer_period = 1/30  # 30 Hz
        self.timer_control = self.create_timer(timer_period, self.control_loop)
        self.timer_pub_angles = self.create_timer(timer_period, self.publish_load_angles) 

    def publish_load_angles(self):
        yaw, pitch, roll = self.get_attitude()
        if yaw is None or pitch is None:
            return
        
        msg = Vector3()
        msg.x = yaw + self.setpoint[0] # add setpoint to get actual angle
        msg.y = pitch + self.setpoint[1] # add setpoint to get actual angle
        msg.z = roll
        self.publish_load_angles_.publish(msg)

    def setpoint_callback(self, msg):
        self.setpoint = [msg.x, msg.y]
        self.get_logger().info(f"Received new setpoint: {self.setpoint[0]:.2f}, {self.setpoint[1]:.2f}")


    def control_loop(self):
        # Set the desired setpoint for yaw and pitch
        yaw_error = self.setpoint[0]
        pitch_error = self.setpoint[1]

        # Wrap angles to [-180, 180]
        wrap_yaw = math.atan2(math.sin(math.radians(yaw_error)), math.cos(math.radians(yaw_error))) * 180 / math.pi
        wrap_pitch = math.atan2(math.sin(math.radians(pitch_error)), math.cos(math.radians(pitch_error))) * 180 / math.pi

        # Update PID controllers
        yaw_output = self.pid_yaw.update(wrap_yaw)
        pitch_output = self.pid_pitch.update(wrap_pitch)

        # Move gimbal with calculated outputs
        self.move_speed(int(min(100, max(-100, yaw_output))), int(min(100, max(-100, pitch_output))))
        #print(f"yaw_speed: {yaw_output: 6.1f}, pitch_speed: {pitch_output: 6.1f}")
        

    def center_gimbal(self):
        # CMD: 0x08  -> center
        msg = b'\x55\x66\x01\x01\x00\x00\x00\x08\x01\xd1\x12'
        while True:
            send(msg)
            response = receive()        
            if response and response[2] == '08' and response[0] == '01':
                print("Gimbal centered")
                return True
            elif response and response[2] == '08' and response[0] == '00':
                print("gimbal failed to center, retrying...")
            time.sleep(0.5)
            print("Gimbal centering failed")

    def move_speed(self, yaw_speed, pitch_speed):
        """
        yaw_speed:   -100 to +100
        pitch_speed: -100 to +100
        """
        # Convert signed to 2 bytes
        ys = yaw_speed & 0xFF
        ps = pitch_speed & 0xFF
        # CMD: 0x07  -> speed control
        msg_front = b'\x55\x66\x01\x02\x00\x00\x00\x07' + bytes([ys, ps])
        self._crc16 = crc16_xmodem(msg_front)
        msg = msg_front + self._crc16
        send(msg)
        response = receive()
        if response and response[2] == '07' and response[0] == '02':
            print(f"Gimbal moving at yaw speed: {yaw_speed}, pitch speed: {pitch_speed}")
            return True
        return False

    def get_attitude(self):
        # CMD: 0x0D -> get attitude
        cmd = bytes.fromhex("556601000000000DE805")
        send(cmd)
        response = receive()
        if response and response[2] == '0d':
            data = response[0]
            # FROMsiyi ros SDK 623:
            #self._att_msg.yaw = toInt(msg[2:4]+msg[0:2]) /10.
            #self._att_msg.pitch = toInt(msg[6:8]+msg[4:6]) /10.
            #self._att_msg.roll = toInt(msg[10:12]+msg[8:10]) /10.
            #self._att_msg.yaw_speed = toInt(msg[14:16]+msg[12:14]) /10.
            #self._att_msg.pitch_speed = toInt(msg[18:20]+msg[16:18]) /10.
            #self._att_msg.roll_speed = toInt(msg[22:24]+msg[20:22]) /10
            yaw = toInt(data[2:4]+data[0:2]) / 10.
            pitch = toInt(data[6:8]+data[4:6]) / 10.
            roll = toInt(data[10:12]+data[8:10]) / 10.
            yaw_speed = toInt(data[14:16]+data[12:14]) / 10.
            pitch_speed = toInt(data[18:20]+data[16:18]) / 10.
            roll_speed = toInt(data[22:24]+data[20:22]) / 10.
            #print(f"Yaw: {yaw: 6.1f}, Pitch: {pitch: 6.1f}, Roll: {roll: 6.1f}, Yaw Speed: {yaw_speed: 6.1f}, Pitch Speed: {pitch_speed: 6.1f}, Roll Speed: {roll_speed: 6.1f}")
            return yaw, pitch, roll
        return None, None, None

        






def main(args=None):
    rclpy.init(args=args)
    gimbal_control = GimbalControlNode()
    rclpy.spin(gimbal_control)
    sock.close()
    gimbal_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
