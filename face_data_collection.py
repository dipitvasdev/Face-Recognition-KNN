import cv2 
import numpy as np 
#Initialize the Camera
cap=cv2.VideoCapture(0)
# Face Dectection
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0 
face_data=[]
dataset_path='./data/'
file_name_=input("Enter your name:")
while True:
	ret,frame=cap.read()
	if ret==False:
		continue #If frame capture fails try again
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
	#Pick the largest face , as it's the largest 
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+h,y+w),(100,200,200),2)
		# Region of Intrest 
		offset=10 
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(200,200))
		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Frame",frame)
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break

# Convert face_data into a numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1)) 
# Save Data into file system
np.save(dataset_path+file_name_+'.npy',face_data)
print("Saved!!")
cap.release()
cv2.destroyAllWindows()