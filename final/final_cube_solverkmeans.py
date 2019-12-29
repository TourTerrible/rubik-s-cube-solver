



import time
import serial



import kociemba
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
cap=cv2.VideoCapture(0)
_,frame=cap.read()
height,width,_=frame.shape


square_size_factor=0.4#fraction of height of image that square occupies
square_x1=int((width-((square_size_factor)*height))/2)#coordinate calculation for square corners
square_x2=int((width+((square_size_factor)*height))/2)


square_y1=int(((1-square_size_factor)/2)*height)
square_y2=int(((1+square_size_factor)/2)*height)

cube_dimension=square_y2-square_y1#height and width of cube in pixels

cube=[]







    
def sol(input1):
	input2=[[[]],[[]],[[]],[[]],[[]],[[]]]

	for i in range(6):
		if(input1[i][1][1]=='w'):
			input2[0]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='r'):
			input2[1]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='g'):
			input2[2]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='y'):
			input2[3]=input1[i]
			break


	for i in range(6):
		if(input1[i][1][1]=='o'):
			input2[4]=input1[i]
			break




	for i in range(6):
		if(input1[i][1][1]=='b'):
			input2[5]=input1[i]
			break

	for i in range(6):
		for j in range(3):
			for k in range(3):
				if (input2[i][j][k]=='w'):
					input2[i][j][k]='U'
				elif (input2[i][j][k]=='y'):
					input2[i][j][k]='D'
				elif (input2[i][j][k]=='r'):
					input2[i][j][k]='R'
				elif (input2[i][j][k]=='o'):
					input2[i][j][k]='L'
				elif (input2[i][j][k]=='g'):
					input2[i][j][k]='F'
				elif(input2[i][j][k]=='b'):
					input2[i][j][k]='B'

	b=''
	for i in range(6):
		for j in range(3):
			for k in range(3):
				b+=input2[i][j][k]
	
       
	a = kociemba.solve(b)
	print(a)
	
	a = a.replace("U'","R L F2 B2 R' L' D' R L F2 B2 R' L'")
	a = a.replace("U2","R L F2 B2 R' L' D2 R L F2 B2 R' L'")
	a = a.replace("U","R L F2 B2 R' L' D R L F2 B2 R' L'")
	a=a.replace("D2","D R R' D")#REMOVE!!!!!!!!!!!!1
	return a

def cleansolution(solution):
    cleanedsolution=""
    prev=solution[0]
    for current in solution[1:]:
        if current=="'":
            cleanedsolution+=prev.lower()
        elif current=="2":
            cleanedsolution+=prev
            cleanedsolution+=prev
        else:
            cleanedsolution+=prev
        prev=current
    cleanedsolution+=solution[len(solution)-1]
    cleanedsolution = cleanedsolution.replace("'", "")
    cleanedsolution = cleanedsolution.replace("2", "")
    cleanedsolution = cleanedsolution.replace(" ", "")
    
    return cleanedsolution

def sendarduino(solution):
    ser = serial.Serial('COM8', 9600,timeout=None) # Establish the connection on a specific port
    time.sleep(1)
    count=0
    for char in solution:
        ser.write(char.encode())
        time.sleep(1)
        #count+=1
        #if count==60:
        #    time.sleep(60)#CHANGE SLEEP TIME ACCORDING TO DELAY BETWEEN MOTORS
        #    count=0
def replace_values(arr,replace,replacement):
    for i in range(len(arr)):
        if arr[i]==replace:
            arr[i]=replacement
    return arr
def find_avg_hsv(img):#given the cropped image of cube face finds the hsv values of each cybie(or tile) and forms a 3d array of hsv values
        
    tile_dimension=int(cube_dimension/3)#as cube face has 9(3x3) tiles
    tile_factor=0.3#factor of area where colour will be found(tile roi)
    tile_roi_start=int(((1-tile_factor)/2)*tile_dimension)#pixels to start of bounding rectangle of tile roi
    tile_roi_end=int((tile_dimension*tile_factor)+tile_roi_start)#pixels to end of bounding rectangle of tile roi
    
    tile_roi=[]#list which will hold roi (in image form) of all individual tiles
    for j in range(3):
        row=[]
        for i in range(3):
            row.append(img[(j*tile_dimension)+tile_roi_start:(j*tile_dimension)+tile_roi_end,(i*tile_dimension)+tile_roi_start:(i*tile_dimension)+tile_roi_end])#roi finding math
            cv2.rectangle(img, ((i*tile_dimension)+tile_roi_start,(j*tile_dimension)+tile_roi_start), ((i*tile_dimension)+tile_roi_end,(j*tile_dimension)+tile_roi_end), (255,0,0), 1)#draws rectangle on each tile roi
        tile_roi.append(row)
    cv2.namedWindow("check")
    cv2.moveWindow("check", 40,30)
    cv2.imshow("check",img)
    hsv_avg=[]#list which will hold avg hsv value of each tile
    #bgr_avg=[]
    #h_all=set()
    #s_all=set()
    #v_all=set()
    for row_iterable in tile_roi:
        row=[]
        bgr_row=[]
        for col_iterable in row_iterable:
            b_avg,g_avg,r_avg,_=np.uint8(cv2.mean(col_iterable))#averages bgr value in roi
            color=cv2.cvtColor(np.uint8([[[b_avg,g_avg,r_avg]]]),cv2.COLOR_BGR2LAB)#converts bgr value to corresponding hsv
            h_avg= color[0][0][0]
            s_avg= color[0][0][1]
            v_avg= color[0][0][2]

            #h_all.add(h_avg)
            #s_all.add(s_avg)
            #v_all.add(v_avg)

            #bgr_row.append([b_avg,g_avg,r_avg])
            row.append([h_avg,s_avg,v_avg])
        hsv_avg.append(row)
    
    return hsv_avg
        #bgr_avg.append(bgr_row)
    #print(hsv_avg)
    
    #print(bgr_avg)
    #print("h",min(h_all),max(h_all))
    #print("s",min(s_all),max(s_all))
    #print("v",min(v_all),max(v_all))
def find_colors(cube):
    cube=np.array(cube).reshape(-1,3)
    
    kmeans=KMeans(n_clusters=6)
    kmeans.fit(cube)
    preds=kmeans.labels_

    preds=preds.tolist()
    unique, counts = np.unique(preds, return_counts=True)


    print(np.asarray((unique, counts)).T)
    if(counts.tolist()!=[9]*6):
        print("COLOR SCAN FAILED")
    else:
        print("SCAN SUCCESFULL")

    preds=replace_values(preds,preds[4],"r")
    preds=replace_values(preds,preds[4+9],"b")
    preds=replace_values(preds,preds[4+9+9],"o")
    preds=replace_values(preds,preds[4+9+9+9],"g")
    preds=replace_values(preds,preds[4+9+9+9+9],"w")
    preds=replace_values(preds,preds[4+9+9+9+9+9],"y")


    preds=np.array(preds).reshape(6,3,3)
    #print(preds)
    return preds
def plot_colors(colors):
    colors=np.where(colors=="r","0",colors)
    colors=np.where(colors=="b","1",colors)
    colors=np.where(colors=="o","2",colors)
    colors=np.where(colors=="g","3",colors)
    colors=np.where(colors=="w","4",colors)
    colors=np.where(colors=="y","5",colors)
    colors=colors.astype(int)
    #print(colors)
    N=12
    data = np.ones((N,N)) * np.nan
    data[3:6,3:6]=colors[0]
    data[3:6,6:9]=colors[1]
    data[3:6,9:12]=colors[2]
    data[3:6,0:3]=colors[3]
    data[0:3,3:6]=colors[4]
    data[6:9,3:6]=colors[5]
    #print(data)
    # make a figure + axes
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # make color map
    my_cmap = matplotlib.colors.ListedColormap(['r', 'b', 'darkorange','g','w','yellow'])
    # set the 'bad' values (nan) to be white and transparent
    my_cmap.set_bad(color='k')
    # draw the grid
    for x in range(N + 1):
        if(x%3==0):
            lw=5
        else:
            lw=2
        ax.axhline(x, lw=lw, color='k', zorder=5)
        ax.axvline(x, lw=lw, color='k', zorder=5)
    # draw the boxes
    ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)
    # turn off the axis labels
    ax.axis('off')

    plt.show()
while True:
    _,frame=cap.read()
    
    cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (255,0,0), 2)#cube should be placed within this square



    cv2.imshow("original",frame)
    k=cv2.waitKey(1) & 0xff
    if k==27:#ESC is pressed
        break
    elif k==32:#SPACE is pressed
        print("Image Captured")
        
        
        cube_roi=frame[square_y1:square_y2,square_x1:square_x2]#image of cube only
        
        cubeface=find_avg_hsv(cube_roi)
        cube.append(cubeface)
        #print("scanned face",cubeface)
        #print("cube",cube)
        
    elif k==ord('s'):
        
        cube_colors=find_colors(cube)
        plot_colors(cube_colors)
        solution=cleansolution(sol(cube))
        print(solution)
        print(len(solution))
        sendarduino(solution)

    elif k==ord('r'):
        cube.pop()





cap.release()
cv2.destroyAllWindows()
