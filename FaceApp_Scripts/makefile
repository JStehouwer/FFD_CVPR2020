all: A10.sh Galaxy.sh MotoG6.sh Pixel.sh Redmi.sh

A10.sh: create_FA_A10
	./create_FA_A10 > A10.sh

Galaxy.sh: create_FA_Galaxy
	./create_FA_Galaxy > Galaxy.sh

MotoG6.sh: create_FA_MotoG6
	./create_FA_MotoG6 > MotoG6.sh

Pixel.sh: create_FA_Pixel
	./create_FA_Pixel > Pixel.sh
	
Redmi.sh: create_FA_Redmi
	./create_FA_Redmi > Redmi.sh


create_FA_A10: create_FA_A10.cpp
	g++ create_FA_A10.cpp -o create_FA_A10 -Wall

create_FA_Galaxy: create_FA_Galaxy.cpp
	g++ create_FA_Galaxy.cpp -o create_FA_Galaxy -Wall

create_FA_MotoG6: create_FA_MotoG6.cpp
	g++ create_FA_MotoG6.cpp -o create_FA_MotoG6 -Wall

create_FA_Pixel: create_FA_Pixel.cpp
	g++ create_FA_Pixel.cpp -o create_FA_Pixel -Wall

create_FA_Redmi: create_FA_Redmi.cpp
	g++ create_FA_Redmi.cpp -o create_FA_Redmi -Wall

clean:
	rm -f create_FA_A10
	rm -f create_FA_Galaxy
	rm -f create_FA_MotoG6
	rm -f create_FA_Pixel
	rm -f create_FA_Redmi
	rm -f A10*.sh
	rm -f Galaxy*.sh
	rm -f MotoG6.sh
	rm -f Pixel*.sh
	rm -f Redmi*.sh
