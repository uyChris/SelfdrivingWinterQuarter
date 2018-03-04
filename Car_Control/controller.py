import pygame
import RPi.GPIO as GPIO
import time
import readchar
import pigpio

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)

#Define pins used.
servoPin = 24
pi = pigpio.pi()


PIN = 18
PWMA1 = 6
PWMA2 = 13
PWMB1 = 20
PWMB2 = 21
PWMC1 = 24
D1 = 12
D2 = 26

PWM = 50

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN,GPIO.IN,GPIO.PUD_UP)
GPIO.setup(PWMA1,GPIO.OUT)
GPIO.setup(PWMA2,GPIO.OUT)
GPIO.setup(PWMB1,GPIO.OUT)
GPIO.setup(PWMB2,GPIO.OUT)
GPIO.setup(PWMC1,GPIO.OUT)
GPIO.setup(D1,GPIO.OUT)
GPIO.setup(D2,GPIO.OUT)
p1 = GPIO.PWM(D1,500)
p2 = GPIO.PWM(D2,500)
p3 = GPIO.PWM(PWMC1,50)
p1.start(0)
p2.start(0)
#p3.start(7.5)

#This function will take a float from -1 to 1 and map it to a degree from 0 to 180
def floattodeg (num):
    # Only allow angle to go from 30 to 150
    # Chassis is blocking wheels from turning anymore
    return (num * 60) + 90

def setAngle(angle):
    angle = int(angle)
    #print("Angle: ", angle)
    pulseWidth = angle*5.55555555555555555555555555555555555556 + 1055
    #print("pulseWidth: ", pulseWidth)
    pi.set_servo_pulsewidth(servoPin,pulseWidth)
    time.sleep(0.05)

#Sets motor output based on four input values through outputting to GPIO pins.
def	set_motor(A1,A2,B1,B2):
	GPIO.output(PWMA1,A1)
	GPIO.output(PWMA2,A2)
	GPIO.output(PWMB1,B1)
	GPIO.output(PWMB2,B2)

#Forward involves pushing both motors forward.
def forward():
    #PWMA1 and PWMB1 set high.
	set_motor(1,0,1,0)

def stop():
	set_motor(0,0,0,0)

def reverse():
    #PWMA2 and PWMB2 set high.
	set_motor(0,1,0,1)

def left():
    #PWMA1 and PWMB2 set high.
	set_motor(1,0,0,0)

def right():
    #PWMA2 and PWMB1 set high.
	set_motor(0,0,1,0)

'''
def getkey():
	if GPIO.input(PIN) == 0:
		count = 0
		while GPIO.input(PIN) == 0 and count < 200:  #9ms
			count += 1
			time.sleep(0.00006)
		count = 0
		while GPIO.input(PIN) == 1 and count < 80:  #4.5ms
			count += 1
			time.sleep(0.00006)
		idx = 0
		cnt = 0
		data = [0,0,0,0]
		for i in range(0,32):
			count = 0
			while GPIO.input(PIN) == 0 and count < 15:    #0.56ms
				count += 1
				time.sleep(0.00006)
			count = 0
			while GPIO.input(PIN) == 1 and count < 40:   #0: 0.56ms
				count += 1                               #1: 1.69ms
				time.sleep(0.00006)
			if count > 8:
				data[idx] |= 1<<cnt
			if cnt == 7:
				cnt = 0
				idx += 1
			else:
				cnt += 1
		if data[0]+data[1] == 0xFF and data[2]+data[3] == 0xFF:  #check
			return data[2]
print('IRM Test Start ...')
stop()
try:
	while True:
		key = readchar.readchar()
		if(key != None):
			print("Get the key: 0x%02x" %key)
			if key == 'w':
				forward()
				print("forward")
			if key == 'a':
				left()
				print("left")
			if key == 's':
				stop()
				print("stop")
			if key == 'd':
				right()
				print("right")
			if key == 'x':
				reverse()
				print("reverse")
			if key == 'e':
				if(PWM + 10 < 101):
					PWM = PWM + 10
					p1.ChangeDutyCycle(PWM)
					p2.ChangeDutyCycle(PWM)
					print(PWM)
			if key == 'q':
				if(PWM - 10 > -1):
					PWM = PWM - 10
					p1.ChangeDutyCycle(PWM)
					p2.ChangeDutyCycle(PWM)
					print(PWM)
except KeyboardInterrupt:
	GPIO.cleanup();
'''
##### ----------------------------------------###
# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def printf(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Set the width and height of the screen [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

#Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

# Get ready to print
textPrint = TextPrint()

# -------- Main Program Loop -----------
while done==False:
    # EVENT PROCESSING STEP
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop

        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")


    # DRAWING STEP
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    textPrint.reset()

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()

    textPrint.printf(screen, "Number of joysticks: {}".format(joystick_count) )
    textPrint.indent()

    #i[0] and i[1] is +/-x and +/-y
    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        textPrint.printf(screen, "Joystick {}".format(i) )
        textPrint.indent()

        # Get the name from the OS for the controller/joystick
        name = joystick.get_name()
        textPrint.printf(screen, "Joystick name: {}".format(name) )

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()
        textPrint.printf(screen, "Number of axes: {}".format(axes) )
        textPrint.indent()

        for i in range( axes ):
            axis = joystick.get_axis( i )
            textPrint.printf(screen, "Axis {} value: {:>6.3f}".format(i, axis) )
        textPrint.unindent()

        buttons = joystick.get_numbuttons()
        textPrint.printf(screen, "Number of buttons: {}".format(buttons) )
        textPrint.indent()

        for i in range( buttons ):
            button = joystick.get_button( i )
            textPrint.printf(screen, "Button {:>2} value: {}".format(i,button) )
        textPrint.unindent()

        # Hat switch. All or nothing for direction, not like joysticks.
        # Value comes back in an array.
        hats = joystick.get_numhats()
        textPrint.printf(screen, "Number of hats: {}".format(hats) )
        textPrint.indent()

        for i in range( hats ):
            hat = joystick.get_hat( i )
            textPrint.printf(screen, "Hat {} value: {}".format(i, str(hat)) )
        textPrint.unindent()

        textPrint.unindent()

        # axis 5 controls the motor pwm and forward and reverse

	# forward/reverse
        #Axis 1 is the up/down axis on the left joystick.
	speed = 40 # speed goes from 0 to 100% power
        if(joystick.get_axis(1) >= 0):

            forward()
            #Currently maxes at 40% power.
            PWM = joystick.get_axis(1)*speed

            p1.ChangeDutyCycle(PWM)
            p2.ChangeDutyCycle(PWM)
            print(PWM)

        elif(joystick.get_axis(1) < 0):

            reverse()
            PWM = abs(joystick.get_axis(1))*speed
            p1.ChangeDutyCycle(PWM)
            p2.ChangeDutyCycle(PWM)
            print(PWM)

        # rotation
        #Axis 2 is the right/left axis on the right joystick.
        if(joystick.get_axis(2) >= 0):
            #Maximum turn from servo is at 10.
            #Servo is centered at 7.5 (car goes straight).
            #PWM = 7.5 + joystick.get_axis(2)*2.5
            #p3.ChangeDutyCycle(PWM)
            #print(PWM)
            # multiply by negative one because car is upside-down
            angle = floattodeg(-1*joystick.get_axis(2))
            setAngle(angle)
            print("Angle: ",angle)
            print("")

        elif(joystick.get_axis(2) < 0):
            #PWM = 7.5 - abs(joystick.get_axis(2))*2.5
            #p3.ChangeDutyCycle(PWM)
            #print(PWM)

            # multiply by negative one because car is upside-down
            angle = floattodeg(-1*joystick.get_axis(2))
            setAngle(angle)
            print("Angle: ",angle)
            print("")
	else:
	    #p3.ChangeDutyCycle(7.5)

            setAngle(90)
            print("Angle: ",angle)
            print("")

    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 20 frames per second
    clock.tick(20)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit ()
