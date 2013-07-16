#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<float.h>
#include<math.h>
#include<vector>

using namespace std;

class Parameter{
    public:
	/***********************************************************************************/
	static const int nFEATURE = 50;
	static const double learningRate = 0.004;//0.0025;//0.0000003;
	static double featureReg;
	static const int MONITER_ITERATION_NUM = 20;


	static const int ID_START_IS_ONE = 1;
	/***********************************************************************************/

	static void dumpParameter(){
		//print setting
		fprintf(stderr, "nFEATURE = %d\n", nFEATURE);
		fprintf(stderr, "learningRate = %lf\n", learningRate);
		fprintf(stderr, "featureReg = %lf\n", featureReg);
		fprintf(stderr, "MONITER_ITERATION_NUM = %d\n", MONITER_ITERATION_NUM);
	}
};


void dealWithID(int& anID){
	if(Parameter::ID_START_IS_ONE){
		assert(anID >= 1);
		--anID;
		return;
	}
	assert(0);
}

inline double pow2(double x){
	return x*x;
}

class RatingList{
    public:
	
	class RatingTuple{
	    public:
		int userID, itemID;
		double rating, originalRating;
		RatingTuple(int a, int b, double c, double d):userID(a),itemID(b),rating(c), originalRating(d){}
	};

	vector<RatingTuple> ratingList;
	int maxUidP1, maxIidP1;
	double maxValue, minValue;
	

	RatingList(const char* fileName = ""){
		if(fileName[0] == '\0'){//dummy RatingList
			maxUidP1 = 0;
			maxIidP1 = 0;
			maxValue = 0;
			minValue = 0;
			return;
		}
		maxUidP1= 0, maxIidP1 = 0;
		FILE* fp = fopen(fileName, "r");
		assert(NULL != fp);
		int u, i;
		double r;
		assert(NULL != fp);
		maxValue = -1000000;	//just a small value
		minValue =  1000000;    //just a large value
		while(fscanf(fp, "%d %d %lf", &u,&i,&r) == 3){
			//start from 1, shift to 0
			dealWithID(u);
			dealWithID(i);

			RatingTuple x(u,i,r,r);
			ratingList.push_back(x);
			maxUidP1 = maxUidP1 >= u+1? maxUidP1: u+1;
			maxIidP1 = maxIidP1 >= i+1? maxIidP1: i+1;

			if(r > maxValue) maxValue = r;
			if(r < minValue) minValue = r;
		}
		assert(-1 == fscanf(fp, "%d", &u));
		fprintf(stderr, "%s: maxUidP1 %d, maxIidP1 %d\n", fileName, maxUidP1, maxIidP1);
		fclose(fp);
	}
};



void getMeanVar_scaleTraining(RatingList& training, double& globalMean, double& globalVariance){
	double squareSum = 0;
	globalMean = 0;
	for(vector<RatingList::RatingTuple>::iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++){
		globalMean += it->rating;
		squareSum += it->rating * it->rating;
	}
	globalMean /= (double)training.ratingList.size();
	globalVariance = sqrt(squareSum/(double)training.ratingList.size() - globalMean*globalMean);
		
	for(vector<RatingList::RatingTuple>::iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++)
		it->rating = (it->rating - globalMean)/globalVariance;
}


class MatrixFactorizationData{
    public:
	static const int nFEATURE = Parameter::nFEATURE;
	RatingList training, validation, testing;
	double globalMean, globalVariance;
	
	const char* trainingName, *validationName, *testingName;
	double (*userFeature)[nFEATURE], (*itemFeature)[nFEATURE], (*userFeatureStep)[nFEATURE], (*itemFeatureStep)[nFEATURE];
	double (*userFeaturePast)[nFEATURE], (*itemFeaturePast)[nFEATURE];
	double (*userBestFeature)[nFEATURE], (*itemBestFeature)[nFEATURE];

	inline void allocateMemory(int maxUidP1, int maxIidP1){
		userFeature = new double[maxUidP1][nFEATURE];
		userFeatureStep = new double[maxUidP1][nFEATURE];
		userFeaturePast = new double[maxUidP1][nFEATURE];
		itemFeature = new double[maxIidP1][nFEATURE];
		itemFeatureStep = new double[maxIidP1][nFEATURE];
		itemFeaturePast = new double[maxIidP1][nFEATURE];
		userBestFeature = new double[maxUidP1][nFEATURE];
		itemBestFeature = new double[maxIidP1][nFEATURE];
	}
	

	void randomInitializeFeature(int maxUidP1, int maxIidP1){
		for(int i = 0; i < maxUidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j){
				userFeature[i][j] = ((double)rand()/RAND_MAX/10 - 0.05)/nFEATURE;
			}
		}
		for(int i = 0; i < maxIidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j){
				itemFeature[i][j] = ((double)rand()/RAND_MAX/10 - 0.05)/nFEATURE;
			}
		}
	}

	MatrixFactorizationData(const char* trainFile, const char* validFile, const char* testFile):training(trainFile), validation(validFile), testing(testFile){
		{
			int i;
			for(i = (int)strlen(trainFile)-1; i >= 0;--i)
				if(trainFile[i] == '/')
					break;
			trainingName = &trainFile[i+1];

			for(i = (int)strlen(validFile)-1; i >= 0;--i)
				if(validFile[i] == '/')
					break;
			validationName = &validFile[i+1];

			for(i = (int)strlen(testFile)-1; i >= 0;--i)
				if(testFile[i] == '/')
					break;
			testingName = &testFile[i+1];
		}

		Parameter:: dumpParameter();

		//some data: last time frame => no data in training
		if(training.maxUidP1 < validation.maxUidP1)
			training.maxUidP1 = validation.maxUidP1;
		if(training.maxUidP1 < testing.maxUidP1)
			training.maxUidP1 = testing.maxUidP1;
		if(training.maxIidP1 < validation.maxIidP1)
			training.maxIidP1 = validation.maxIidP1;
		if(training.maxIidP1 < testing.maxIidP1)
			training.maxIidP1 = testing.maxIidP1;
	
		fprintf(stderr, "train: maxUidP1 %d maxIidP1 %d\n", training.maxUidP1, training.maxIidP1);
		fprintf(stderr, "valid: maxUidP1 %d maxIidP1 %d\n", validation.maxUidP1, validation.maxIidP1);
		fprintf(stderr, "test: maxUidP1 %d maxIidP1 %d\n", testing.maxUidP1, testing.maxIidP1);
		fprintf(stderr, "train: max%lf min%lf\n", training.maxValue, training.minValue);
		fprintf(stderr, "valid: max%lf min%lf\n", validation.maxValue, validation.minValue);
		fprintf(stderr, "test: max%lf min%lf\n", testing.maxValue, testing.minValue);

//		assert(training.maxValue == validation.maxValue && training.minValue == validation.minValue);
//		assert(training.maxValue == testing.maxValue && training.minValue == testing.minValue);
		assert(training.maxUidP1 >= validation.maxUidP1 && training.maxIidP1 >= validation.maxIidP1);
		assert(training.maxUidP1 >= testing.maxUidP1 && training.maxIidP1 >= testing.maxIidP1);
	
		allocateMemory(training.maxUidP1, training.maxIidP1);
		
		globalMean = 0, globalVariance = 1;
		getMeanVar_scaleTraining(training, globalMean, globalVariance);
		fprintf(stderr, "globalMean = %lf, globalVariance = %lf\n", globalMean, globalVariance);

		randomInitializeFeature(training.maxUidP1, training.maxIidP1);
	}


	void calculateGeneralError(const RatingList& inList, double& theRMSE, double& theMAE, const double meanOfModel, const double varianceOfModel){
		theRMSE = 0;
		theMAE = 0;
		for(vector<RatingList::RatingTuple>::const_iterator it = inList.ratingList.begin(); it != inList.ratingList.end(); it++){
			int userID = it->userID, itemID = it->itemID;
			double epsilon = 0;
			for(int j = 0; j < nFEATURE; ++j)
				epsilon += userFeature[userID][j] * itemFeature[itemID][j];
	
			epsilon *= varianceOfModel;
			epsilon += meanOfModel - it->rating;

			//clipped penalty
/*			if(it->originalRating == inList.maxValue && epsilon > 0)
				epsilon = 0;
			if(it->originalRating == inList.minValue && epsilon < 0)
				epsilon = 0;*/
			
			theRMSE += epsilon*epsilon;
			theMAE += fabs(epsilon);
		}
		theRMSE = sqrt(theRMSE/(double)inList.ratingList.size());
		theMAE /= (double)inList.ratingList.size();
	}

	void calculateTrainingError(double& theRMSE, double& theMAE, double& objectiveValue){
		objectiveValue = 0;
		for(int userID = 0; userID < training.maxUidP1; ++userID)
			for(int k = 0; k < nFEATURE; ++k)
				objectiveValue += pow2(userFeature[userID][k]);
		for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
			for(int k = 0; k < nFEATURE; ++k)
				objectiveValue += pow2(itemFeature[itemID][k]);
		objectiveValue *= Parameter::featureReg;
		calculateGeneralError(training, theRMSE, theMAE, 0, 1);
		objectiveValue += theRMSE * theRMSE * (double)training.ratingList.size();
		theRMSE *= globalVariance;
		theMAE *= globalVariance;
	}
	void calculateValidError(double& theRMSE, double& theMAE){
		calculateGeneralError(validation, theRMSE, theMAE, globalMean, globalVariance);
	}
	void calculateTestError(double& theRMSE, double& theMAE){
		calculateGeneralError(testing, theRMSE, theMAE, globalMean, globalVariance);
	}

	void dumpPrediction(){
		char tmpName[100];
		sprintf(tmpName, "pred_%s", validationName);
		FILE *validPred = fopen(tmpName, "w");
		for(vector<RatingList::RatingTuple>::iterator it = validation.ratingList.begin(); it != validation.ratingList.end(); it++){
			double pred = 0;
			for(int k = 0; k < nFEATURE; ++k)
				pred += userFeature[it->userID][k]*itemFeature[it->itemID][k];
			pred = pred*globalVariance + globalMean;
			fprintf(validPred, "%lf\n", pred);
		}
		fclose(validPred);


		sprintf(tmpName, "pred_%s", testingName);
		FILE *testPred = fopen(tmpName, "w");
		for(vector<RatingList::RatingTuple>::iterator it = testing.ratingList.begin(); it != testing.ratingList.end(); it++){
			double pred = 0;
			for(int k = 0; k < nFEATURE; ++k)
				pred += userFeature[it->userID][k]*itemFeature[it->itemID][k];
			pred = pred*globalVariance + globalMean;
			fprintf(testPred, "%lf\n", pred);
		}
		fclose(testPred);
	}

	void dumpModel(const char* modelName){
		FILE *model = fopen(modelName, "w");
		fprintf(model, "UserLatent\n");
		for(int i = 0; i < training.maxUidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				fprintf(model, " %lf", userFeature[i][j]);
			fprintf(model, "\n");
		}
		fprintf(model, "ItemLatent\n");
		for(int i = 0; i < training.maxIidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				fprintf(model, " %lf", itemFeature[i][j]);
			fprintf(model, "\n");
		}
		fclose(model);
	}
};


class MatrixFactorization{
    public:
	/***********************************************************************************/
	static const int nFEATURE = Parameter::nFEATURE;
	static const int MONITER_ITERATION_NUM = Parameter::MONITER_ITERATION_NUM;
	/***********************************************************************************/


	static void updateModel(MatrixFactorizationData& mfData){
		const RatingList& training = mfData.training;
		const double learningRate = Parameter::learningRate;
		const double featureReg = Parameter::featureReg;
		for(int userID = 0; userID < training.maxUidP1; ++userID)
			for(int k = 0; k < nFEATURE; ++k)
				mfData.userFeatureStep[userID][k] = -featureReg*mfData.userFeature[userID][k];
		for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
			for(int k = 0; k < nFEATURE; ++k)
				mfData.itemFeatureStep[itemID][k] = -featureReg*mfData.itemFeature[itemID][k];

		for(vector<RatingList::RatingTuple>::const_iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++){
			int userID = it->userID, itemID = it->itemID;
			double rating = it->rating;
			double epsilon = -rating;
			for(int k = 0; k < nFEATURE; ++k)
				epsilon += mfData.userFeature[userID][k] * mfData.itemFeature[itemID][k];
			for(int k = 0; k < nFEATURE; ++k){
				double tmp1 = mfData.userFeature[userID][k], tmp2 = mfData.itemFeature[itemID][k];
				mfData.userFeatureStep[userID][k] -= epsilon * tmp2;
				mfData.itemFeatureStep[itemID][k] -= epsilon * tmp1;
			}
		}

		double preRMSE, preMAE, preObjectiveValue;
		mfData.calculateTrainingError(preRMSE, preMAE, preObjectiveValue);
		double learningRateNow = learningRate;

		for(int userID = 0; userID < training.maxUidP1; ++userID){
			for(int k = 0; k < nFEATURE; ++k)
				mfData.userFeaturePast[userID][k] = mfData.userFeature[userID][k];
		}
		for(int itemID = 0; itemID < training.maxIidP1; ++itemID){
			for(int k = 0; k < nFEATURE; ++k)
				mfData.itemFeaturePast[itemID][k] = mfData.itemFeature[itemID][k];
		}

		//line search
		while(1){
			for(int userID = 0; userID < training.maxUidP1; ++userID){
				for(int k = 0; k < nFEATURE; ++k){
					assert(!isnan((float)mfData.userFeatureStep[userID][k]));
					mfData.userFeature[userID][k] = mfData.userFeaturePast[userID][k] + learningRateNow * mfData.userFeatureStep[userID][k];
				}
			}
			for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
				for(int k = 0; k < nFEATURE; ++k){
					assert(!isnan((float)mfData.itemFeatureStep[itemID][k]));
					mfData.itemFeature[itemID][k] = mfData.itemFeaturePast[itemID][k] + learningRateNow * mfData.itemFeatureStep[itemID][k];
				}

			double nowRMSE, nowMAE, nowObjectiveValue;
			mfData.calculateTrainingError(nowRMSE, nowMAE, nowObjectiveValue);
			if(nowObjectiveValue <= preObjectiveValue)
				break;
			learningRateNow /= 2;
			fprintf(stderr, " preRMSE %lf nowRMSE %lf, preObjectiveValue %lf, nowObjectiveValue %lf, learningRateNow = %lf\n", preRMSE, nowRMSE, preObjectiveValue, nowObjectiveValue, learningRateNow);
		}

	}
/*
	destructor
*/

	static int startOptimizating(MatrixFactorizationData& mfData, int fixedIter){	
		double trainingRMSE, validationRMSE, testingRMSE, trainingMAE, validationMAE, testingMAE, preValidationRMSE = DBL_MAX, bestValidationRMSE = DBL_MAX, objectiveValue;
		int iter = 1, bestIter = -1;

		while(1){
			//calculate all 
			mfData.calculateTrainingError(trainingRMSE, trainingMAE, objectiveValue);
			mfData.calculateValidError(validationRMSE, validationMAE);
			mfData.calculateTestError(testingRMSE, testingMAE);
			fprintf(stderr, "currentIter = %d, bestIter = %d\n", iter, bestIter);
			fprintf(stderr, "trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", trainingRMSE, trainingMAE, validationRMSE, validationMAE, testingRMSE, testingMAE);
			
			//save the currently best model
			if(bestValidationRMSE > validationRMSE){
				bestValidationRMSE = validationRMSE;
				bestIter = iter;
				for(int i = 0; i < mfData.training.maxUidP1; ++i){
					for(int j = 0; j < nFEATURE; ++j)
						mfData.userBestFeature[i][j] = mfData.userFeature[i][j];
				}
				for(int i = 0; i < mfData.training.maxIidP1; ++i){
					for(int j = 0; j < nFEATURE; ++j)
						mfData.itemBestFeature[i][j] = mfData.itemFeature[i][j];
				}
			}

			//stop criterion
			if(fixedIter == 0){	//look at validation to stop
				if(preValidationRMSE - 1e-7 <= validationRMSE && iter > (bestIter+MONITER_ITERATION_NUM))
					break;
			}
			else{
				if(iter == fixedIter)
					break;
			}

			preValidationRMSE = validationRMSE;
			
			updateModel(mfData);

			++iter;
		}
	
		
		fprintf(stderr, "end iter: %d, bestIter: %d\n", iter, bestIter);
		if(fixedIter != 0){
			mfData.calculateTrainingError(trainingRMSE, trainingMAE, objectiveValue);
			mfData.calculateValidError(validationRMSE, validationMAE);
			mfData.calculateTestError(testingRMSE, testingMAE);
			fprintf(stderr, "end Iter: trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", trainingRMSE, trainingMAE, validationRMSE, validationMAE, testingRMSE, testingMAE);
		}
	
		if(fixedIter == 0){//not the retrain, in retrain we can't look at testing RMSE to decide our model
			//copy the best mode back
			for(int i = 0; i < mfData.training.maxUidP1; ++i){
				for(int j = 0; j < nFEATURE; ++j)
					mfData.userFeature[i][j] = mfData.userBestFeature[i][j];
			}
			for(int i = 0; i < mfData.training.maxIidP1; ++i){
				for(int j = 0; j < nFEATURE; ++j)
					mfData.itemFeature[i][j] = mfData.itemBestFeature[i][j];
			}
		

			for(int k = 0; k < nFEATURE; ++k){
				double userLen = 0, itemLen = 0;
//				for(int userID = 0; userID < training.maxUidP1; ++userID)
//					if(fabs(userFeature[userID][k]) > userLen)
//						userLen = fabs(userFeature[userID][k]);
//				for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
//					if(fabs(itemFeature[itemID][k]) > itemLen)
//						itemLen = fabs(itemFeature[itemID][k]);
				for(int userID = 0; userID < mfData.training.maxUidP1; ++userID)
					userLen += mfData.userFeature[userID][k] * mfData.userFeature[userID][k];
				for(int itemID = 0; itemID < mfData.training.maxIidP1; ++itemID)
					itemLen += mfData.itemFeature[itemID][k] * mfData.itemFeature[itemID][k];
				fprintf(stderr, "%lf\n", sqrt(userLen * itemLen));
			}

			mfData.calculateTrainingError(trainingRMSE, trainingMAE, objectiveValue);
			mfData.calculateValidError(validationRMSE, validationMAE);
			mfData.calculateTestError(testingRMSE, testingMAE);
			fprintf(stderr, "reg = %lf, bestIter: trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", Parameter::featureReg, trainingRMSE, trainingMAE, validationRMSE, validationMAE, testingRMSE, testingMAE);
		}

		char modelName[1000];
		if(fixedIter == 0)
			sprintf(modelName, "%s_vali_model.txt", mfData.trainingName);
		else
			sprintf(modelName, "%s_retrain_model.txt", mfData.trainingName);
		
			
		mfData.dumpModel(modelName);
		mfData.dumpPrediction();
		return bestIter;
	}
	
};


double Parameter::featureReg;
int main(int argc, char* argv[]){
	srand((unsigned)time(NULL));
	assert(argc == 5);
	Parameter::featureReg = atof(argv[1]);

        const char *train = argv[2], *valid = argv[3], *test = argv[4];

	fprintf(stderr, "valid To stop\n");
	MatrixFactorizationData mfData(train, valid, test);
	MatrixFactorization::startOptimizating(mfData, 0);
}

