"""HSV VALUES
WILL VARY SLIGHTLY WITH RUBIKS CUBE,CAMERA AND LIGHTING CONDITIONS
Red:    H 165-180
        S 200-255
        V 0-255
Orange: H 0-9
        S 200-255
        V 0-255
White:  H 100-130
        S 0-100
        V 0-255
Yellow: H 10-35
        S 100-255
        V 0-255
Blue:   H 110-120
        S 200-255
        V 0-255
Green   H 60-109
        S 200-255
        V 0-255
As you may notice white and blue have intersecting blue values but vary in their saturation values
If lighting is not proper orange may be interpreted as red or vice versa
"""

#hsv ranges in list form [[min_h,min_s,min_v],[max_h,max_s,max_v]]
red_hsv=[[150,200,0],[180,255,255]]
orange_hsv=[[0,200,0],[9,255,255]]
white_hsv=[[100,0,0],[130,199,255]]
yellow_hsv=[[10,100,0],[39,255,255]]
blue_hsv=[[110,200,0],[149,255,255]]
green_hsv=[[40,200,0],[109,255,255]]








import cv2
import numpy as np
cap=cv2.VideoCapture(0)
_,frame=cap.read()
height,width,_=frame.shape


square_size_factor=0.4#fraction of height of image that square occupies
square_x1=int((width-((square_size_factor)*height))/2)#coordinate calculation for square corners
square_x2=int((width+((square_size_factor)*height))/2)


square_y1=int(((1-square_size_factor)/2)*height)
square_y2=int(((1+square_size_factor)/2)*height)

cube_dimension=square_y2-square_y1#height and width of cube in pixels


def find_color(h,s,v):#finds color given h,s,v values(average)
    if red_hsv[0][0]<=h<=red_hsv[1][0]:
        return "r"
    elif orange_hsv[0][0]<=h<=orange_hsv[1][0]:
        return "o"
    elif yellow_hsv[0][0]<=h<=yellow_hsv[1][0]:
        return "y"
    elif green_hsv[0][0]<=h<=green_hsv[1][0]:
        return "g"
    elif white_hsv[0][0]<=h<=white_hsv[1][0] and white_hsv[0][1]<=s<=white_hsv[1][1]:
        return "w"
    elif blue_hsv[0][0]<=h<=white_hsv[1][0] and blue_hsv[0][1]<=s<=blue_hsv[1][1]:
        return "b"    
    else:
        print("h ",h," s ",s," v ",v,"colour not recognised")


def find_face_colors(face_hsv):#finds color of each tile and forms a 2d array of the face
    face_colors=[]
    for row in face_hsv:
        row_colors=[]
        for tile in row:
            
            row_colors.append(find_color(tile[0],tile[1],tile[2]))
        face_colors.append(row_colors)
    return face_colors



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
            color=cv2.cvtColor(np.uint8([[[b_avg,g_avg,r_avg]]]),cv2.COLOR_BGR2HSV)#converts bgr value to corresponding hsv
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
    

    

while True:
    
    
    _,frame=cap.read()
    frame=cv2.GaussianBlur(frame,(5,5),0)
    edges = cv2.Canny(frame, 80, 250)


    img2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    image = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True) #Removes tiny deviations, tries to go for straight lines.
        if len(approx) == 4: #Lines, should be square.
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 150 and w< 210 and h > 160 and h <240:
                print("Width and height: ", w, h)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                image = frame[y+5:y+h+5,x+5:x+w+5]
                print(find_face_colors(find_avg_hsv(image)))
                break
            
                """if k==32:#space is pressed
                    image = frame[y:y+h,x:x+w]
                    print(find_face_colors(find_avg_hsv(image)))"""
                
                
    
    
    
    cv2.imshow('edges', edges)
    cv2.imshow('area', image)
    cv2.imshow("original",frame)
    k=cv2.waitKey(1) & 0xff
    if k==27:#ESC is pressed
        break
    """elif k==32:#SPACE is pressed
        print("Image Captured")
        
        
        #cube_roi=frame[square_y1:square_y2,square_x1:square_x2]#image of cube only
        

        
         print(find_face_colors(find_avg_hsv(image)))#calls all the functions"""
    





cap.release()
cv2.destroyAllWindows()