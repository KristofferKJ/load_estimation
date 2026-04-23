import gpiod
import time

chip = gpiod.Chip('gpiochip4')  # This is where GPIO17 lives
line = chip.get_line(17)

line.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        line.set_value(1)
        print("ON")
        time.sleep(1)

        line.set_value(0)
        print("OFF")
        time.sleep(1)

except KeyboardInterrupt:
    line.set_value(0)
    line.release()
