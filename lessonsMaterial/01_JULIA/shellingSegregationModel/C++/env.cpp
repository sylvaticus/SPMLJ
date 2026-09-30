#include <algorithm>
//#include <iostream>
#include <fstream>
#include "env.h"

env::env()
{
    //ctor
}

env::~env()
{
    //dtor
    for (int i=0;i<agents.size();i++){
        delete agents.at(i);
    }

}



void
env::init(){
    getParameters();
    createEnvironment();
    randomlyLocateAgents();
}

void
env::getParameters(){

    cout << "Inserisci i parametri.." << endl;
    cout << "Numero di righe:" << endl;
    rows = askInteger(100,30);
    cout << "Numero di colonne:" << endl;
    cols = askInteger(100,30);
    cout << "Numero di agenti del primo tipo:" << endl;
    nAg1 = askInteger((rows*cols)*0.8, (rows*cols)*0.33);
    cout << "Numero di agenti del secondo tipo:" << endl;
    nAg2 = askInteger((rows*cols)-nAg1-2, ((rows*cols)-nAg1)*0.33);
    cout << "Nome degli agenti del primo tipo:" << endl;
    nameAg1 = askString("Locali");
    cout << "Nome degli agenti del secondo tipo:" << endl;
    nameAg1 = askString("Stranieri");
    cout << "Happiness Threshold:" << endl;
    happinessThreshold = askFloat(0.8, 0.4);
    cout << "How far looks for similar neighbouroughs (concentric rings) ?" << endl;
    watchRings = askInteger(rows/2, rows/12);
    cout << "How many steps this model should ring for ?" << endl;
    totalSteps = askInteger(9999,10);
    cout << "Ogni quanti steps visualizzare/stampare lo stato corrente del modello ?" << endl;
    displaySteps = askInteger(200,1);
    cout << "Quanti secondi aspettare tra uno step e l'altro?" << endl;
    waitSeconds = askInteger(4, 1);
    cout << "Inserimento parametri completato." << endl;
}

void
env::createEnvironment(){
    for (int i=0;i<nAg1;i++){
        agent* AG = new agent();
        AG->setType(1);
        AG->setENV(this);
        agents.push_back(AG);
    }
    for (int i=0;i<nAg2;i++){
        agent* AG = new agent();
        AG->setType(2);
        AG->setENV(this);
        agents.push_back(AG);
    }
    vector <int> tempVector(rows*cols, 0);
    cachedPixels=tempVector;
}

void
env::randomlyLocateAgents(){
   for (int i=0;i<agents.size();i++){
        for (;;){
            int randomPosition = int (0+( (double)rand() / ((double)(RAND_MAX)+(double)(1)) )*(rows*cols-0)); // randomRow is [0,nrows*cols[
            if(cachedPixels[randomPosition] == 0){
                cachedPixels[randomPosition] = agents[i]->getType();
                int X = posToX(randomPosition);
                agents[i]->setHomeX(X);
                agents[i]->setHomeY(posToY(randomPosition));
                break;
            }
        }
   }
}

void
env::step(int stepn){
   for (int i=0;i<agents.size();i++){
        if(agents[i]->isHappy()) {
            continue;
        } else {
            agents[i]->move();
        }
   }
}

void
env::print(int stepn){
    if(stepn != totalSteps-1) { // to allow priting of the last step
        if(stepn%displaySteps) return; // not printing intermediate steps
    }
    cout << "Step: " << stepn << endl;
    for(int i=0;i<rows;i++){
        for (int y=0;y<cols;y++){
            int pos = xyToPos(y,i); // attenction here!
            if (!cachedPixels[pos]) {
                cout << ". ";
            } else {
                cout << cachedPixels[pos] << " ";
            }
        }
    cout << endl;
  }

}


int
env::getNeighbors(int x, int y){
    int totalNeighbors = 0;
    totalNeighbors += getNeighbors(x, y, 1);
    totalNeighbors += getNeighbors(x, y, 2);
    return totalNeighbors;
}

int
env::getNeighbors(int x, int y, int type){
    int neighbors = 0;
    int rings = watchRings;
    for(int i=0; i<cachedPixels.size(); i++){
        int xpx = posToX(i);
        int ypx = posToY(i);
      if(
         xpx>= x-watchRings && xpx <= x+watchRings  && ypx >= y-watchRings && ypx <= y+watchRings
         &&
         (xpx != x || ypx != y) // I don't want count the calling agent as neighbors
         &&
         cachedPixels[i] == type
         ) {
         neighbors ++;
      }
    }
    return neighbors;
}


void
env::refreshCachedPixels(){
    vector <int> tempVector(rows*cols, 0);
    cachedPixels=tempVector;
    for (int i=0;i<agents.size();i++){
        int pos = xyToPos(agents[i]->getHomeX(),agents[i]->getHomeY());
        cachedPixels[pos] = agents[i]->getType();
    }
}

vector<int>
env::getEmptyPositions(){
    vector <int> toReturn;
    for(int i=0;i<cachedPixels.size();i++){
        if ( cachedPixels.at(i)==0) {
            toReturn.push_back(i);
        }
    }
    random_shuffle(toReturn.begin(), toReturn.end());
    return toReturn;
}


int
env::askInteger(int maxvalue, int defaultValue){
  string tempStringValue;
  int tempIntegerValue;
  cout << "Inserisci un valore inferiore a " << maxvalue << "(default: " << defaultValue << ")" << endl;
  for(;;){ // ciclo infinito
    getline(cin, tempStringValue);
    tempIntegerValue = s2i(tempStringValue);
    if (tempStringValue.empty()) {
       return defaultValue;
    } else if ( tempIntegerValue <= maxvalue){
       return tempIntegerValue;
    }
    else cout << "Devi fornire un valore inferiore a " << maxvalue << " !!" << endl;
  }
}

float
env::askFloat(float maxvalue, float defaultValue){
  string tempStringValue;
  float tempFloatValue;
  cout << "Inserisci un valore inferiore a " << maxvalue << "(default: " << defaultValue << ")" << endl;
  for(;;){ // ciclo infinito
    getline(cin, tempStringValue);
    tempFloatValue = s2f(tempStringValue);
    if (tempStringValue.empty()) {
       return defaultValue;
    } else if ( tempFloatValue <= maxvalue){
       return tempFloatValue;
    }
    else cout << "Devi fornire un valore inferiore a " << maxvalue << " !!" << endl;
  }
  return defaultValue;
}


string
env::askString(string defaultValue){
  string tempStringValue;
  cout << "Inserisci una stringa (default: " << defaultValue << ")" << endl;
  getline(cin, tempStringValue);
  if (tempStringValue.empty()) {
     return defaultValue;
  } else {
     return tempStringValue;
  }
}


int
env::s2i ( string string_h) const {
	if (string_h == "") return 0;
	int valueAsInteger;
	string valueAsString = string_h;
	istringstream totalSString( valueAsString );
	totalSString >> valueAsInteger;
	return valueAsInteger;
}

/// Includes comma to dot conversion if needed.
float
env::s2f ( string string_h) const {
	if (string_h == "") return 0;
	float valueAsFloat;
	string valueAsString = string_h;
	// replace commas with dots.
	replace(valueAsString.begin(), valueAsString.end(), ',', '.');
	istringstream totalSString( valueAsString );
	totalSString >> valueAsFloat;
	return valueAsFloat;
}
/// Includes conversion checks.
bool
env::s2b ( string string_h) const {
	if (string_h == "true" || string_h == "vero" || string_h == "TRUE" || string_h == "1")
		return true;
	else if (string_h == "false" || string_h == "falso" || string_h == "FALSE" || string_h == "0")
		return false;

	cout << "ERROR: Sorry, I don't know how to convert " << string_h << " to a bool value. I return true... hope for the best" << endl;
	return true;
}

string
env::i2s (int int_h) const{
	ostringstream out;
	out<<int_h;
	return out.str();
}

string
env::d2s (double double_h) const{
	ostringstream out;
	out<<double_h;
	return out.str();
}

string
env::b2s (bool bool_h) const{
	string out;
	if(bool_h)
		out = "true";
	else
		out = "false";
	return out;
}



