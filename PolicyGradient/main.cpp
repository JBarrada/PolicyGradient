#include <glew.h>
#include <glut.h>

#include <vector>
#include <cmath>
#include <numeric>

#include "glm\glm.hpp"
#include "glm\gtx\rotate_vector.hpp"

#include "Box2D\Box2D.h"

#include "NN.h"

#include <iostream>
#include <fstream>

using namespace std;

#define M_PI 3.14159265359f

int SCREEN_W = 320;
int SCREEN_H = 240;
float ASPECT = (float)SCREEN_W / (float)SCREEN_H;

float framerate = 30.0f;

float view_scale = 10.0f;

float32 timeStep = 1.0f / 30.0f;
int32 velocityIterations = 6;
int32 positionIterations = 2;

b2Vec2 gravity(0.0f, -1.0f);
b2World world(gravity);

b2Body* body;

float boosters_vals[2] = { 0, 0 };

vector<unsigned> topology = { 6, 9, 4, 2 };
//vector<unsigned> topology = { 6, 12, 12, 9, 3, 2 };
Net myNet(topology);

int current_time = 0;
int max_time = 1000;

struct training_data_entry {
	float angle, anvel, horzvel, vertvel, targetdx, targetdy;
	float boost_left, boost_right;

	float score_delta;
};

vector<training_data_entry> training_data;

vector<training_data_entry> self_training_data;

void init_world() {
	//b2BodyDef groundBodyDef;
	//groundBodyDef.position.Set(0.0f, -10.0f);

	// Call the body factory which allocates memory for the ground body
	// from a pool and creates the ground box shape (also from a pool).
	// The body is also added to the world.
	//b2Body* groundBody = world.CreateBody(&groundBodyDef);

	// Define the ground box shape.
	//b2PolygonShape groundBox;

	// The extents are the half-widths of the box.
	//groundBox.SetAsBox(50.0f, 10.0f);

	// Add the ground fixture to the ground body.
	//groundBody->CreateFixture(&groundBox, 0.0f);

	// Define the dynamic body. We set its position and call the body factory.
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;
	bodyDef.position.Set(16.0, 12.0);
	body = world.CreateBody(&bodyDef);

	// Define another box shape for our dynamic body.
	b2PolygonShape dynamicBox;
	dynamicBox.SetAsBox(1.0f, 1.0f);

	// Define the dynamic body fixture.
	b2FixtureDef fixtureDef;
	fixtureDef.shape = &dynamicBox;

	// Set the box density to be non-zero, so it will be dynamic.
	fixtureDef.density = 1.0f;

	// Override the default friction.
	fixtureDef.friction = 0.3f;

	// Add the shape to the body.
	body->CreateFixture(&fixtureDef);

	//body->ApplyForceToCenter(b2Vec2(500, 500), true);
	//body->ApplyAngularImpulse(1.0f, true);
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT);

	const b2Transform& xf = body->GetTransform();
	for (b2Fixture* f = body->GetFixtureList(); f; f = f->GetNext()) {
		b2PolygonShape* poly = (b2PolygonShape*)f->GetShape();

		glBegin(GL_POLYGON);
		glColor3f(0.08f, 0.08f, 0.08f);
		for (int i = 0; i < poly->m_count; i++) {
			b2Vec2 current_vertex = b2Mul(xf, poly->m_vertices[i]);
			glVertex2f(current_vertex.x * view_scale, current_vertex.y * view_scale);
		}
		glEnd();
	}


	b2Vec2 box_pos = body->GetPosition();

	glm::vec2 box_bl = glm::rotate(glm::vec2(-1.0f, -1.0f), body->GetAngle());
	glm::vec2 box_br = glm::rotate(glm::vec2(1.0f, -1.0f), body->GetAngle());
	glm::vec2 box_tr = glm::rotate(glm::vec2(1.0f, 1.0f), body->GetAngle());
	glm::vec2 box_tl = glm::rotate(glm::vec2(-1.0f, 1.0f), body->GetAngle());


	float center_x = SCREEN_W / 2.0f;
	float center_y = SCREEN_H / 2.0f;

	glBegin(GL_QUADS);
	glColor3f(boosters_vals[0], 0.0f, 0.1f);
	glVertex2f(center_x + (box_bl.x * view_scale), center_y + (box_bl.y * view_scale));
	glColor3f(boosters_vals[1], 0.0f, 0.1f);
	glVertex2f(center_x + (box_br.x * view_scale), center_y + (box_br.y * view_scale));
	glColor3f(0.0f, 0.08f, 0.0f);
	glVertex2f(center_x + (box_tr.x * view_scale), center_y + (box_tr.y * view_scale));
	glVertex2f(center_x + (box_tl.x * view_scale), center_y + (box_tl.y * view_scale));
	glEnd();

	glutSwapBuffers();
}

void keyboard_down_special(int key, int x, int y) {

}

void keyboard_up_special(int key, int x, int y) {

}

void keyboard_down(unsigned char key, int x, int y) {
	if (key == 'p') {
		b2Vec2 randomImpulse(((rand() / (double)(RAND_MAX)) * 4.0) - 2.0, ((rand() / (double)(RAND_MAX)) * 4.0) - 2.0);
		body->ApplyLinearImpulseToCenter(randomImpulse, true);
		double random_val = (((rand() / (double)(RAND_MAX)) * 1.0) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
		body->SetAngularVelocity(random_val);
	}
}

void keyboard_up(unsigned char key, int x, int y) {
	if (key == ' ') {
		body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
		body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

		// rnd ang vel
		//double random_val = (((rand() / (double)(RAND_MAX)) * 0.5) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
		//body->SetAngularVelocity(random_val);


		//float newpos_x = ((rand() / (float)RAND_MAX) * SCREEN_W) / view_scale;
		//float newpos_y = ((rand() / (float)RAND_MAX) * SCREEN_H) / view_scale;

		float rand_angle = (rand() / (double)(RAND_MAX)) * M_PI * 2.0f;
		float rand_radius = ((rand() / (double)(RAND_MAX)) * ((SCREEN_H / 4.0f) / view_scale)) + ((SCREEN_H / 4.0f) / view_scale);
		//rand_radius /= 2.0f;

		float newpos_x = glm::cos(rand_angle) * rand_radius + ((SCREEN_W / 2.0f) / view_scale);
		float newpos_y = glm::sin(rand_angle) * rand_radius + ((SCREEN_H / 2.0f) / view_scale);


		body->SetTransform(b2Vec2(newpos_x, newpos_y), 0.0f);
	}
}


void boosters(float left, float right) {
	left = glm::clamp(left, 0.0f, 1.0f);

	//if (left < 0.9) left = 0.0; // debug

	b2Vec2 left_force_dir = body->GetWorldVector(b2Vec2(0.0f, left * 5.0f));
	b2Vec2 left_force_point = body->GetWorldPoint(b2Vec2(-1.0f, 0.0f));

	body->ApplyForce(left_force_dir, left_force_point, true);

	right = glm::clamp(right, 0.0f, 1.0f);

	//if (right < 0.9) right = 0.0; // debug

	b2Vec2 right_force_dir = body->GetWorldVector(b2Vec2(0.0f, right * 5.0f));
	b2Vec2 right_force_point = body->GetWorldPoint(b2Vec2(1.0f, 0.0f));

	body->ApplyForce(right_force_dir, right_force_point, true);

	boosters_vals[0] = left;
	boosters_vals[1] = right;
}

void force_redraw(int value) {
	glutPostRedisplay();

	// sig 0 -> 0.5 -> 1
	float input_angle = glm::mod((float)body->GetAngle() - M_PI, M_PI * 2.0f) / (M_PI * 2.0f);
	float input_anvel = glm::clamp(body->GetAngularVelocity() + 2.0f, 0.0f, 4.0f) / 4.0f;
	float input_horzvel = glm::clamp(body->GetLinearVelocity().x + 2.0f, 0.0f, 4.0f) / 4.0f;
	float input_vertvel = glm::clamp(body->GetLinearVelocity().y + 2.0f, 0.0f, 4.0f) / 4.0f;

	b2Vec2 pos = body->GetPosition();
	b2Vec2 target(16.0f, 12.0f);
	b2Vec2 target_delta = (target - pos);
	float input_posx = (target_delta.x * view_scale) / (float)SCREEN_W * 2.0;
	float input_posy = (target_delta.y * view_scale) / (float)SCREEN_H * 2.0;
	input_posx = glm::clamp(input_posx + 1.0f, 0.0f, 2.0f) / 2.0f;
	input_posy = glm::clamp(input_posy + 1.0f, 0.0f, 2.0f) / 2.0f;

	vector<double> net_inputs;
	vector<double> net_outputs;


	// tanh -1 -> 1
	input_angle = (input_angle * 2.0) - 1.0;
	input_anvel = (input_anvel * 2.0) - 1.0;
	input_horzvel = (input_horzvel * 2.0) - 1.0;
	input_vertvel = (input_vertvel * 2.0) - 1.0;
	input_posx = (input_posx * 2.0) - 1.0;
	input_posy = (input_posy * 2.0) - 1.0;

	//squarert everythign
	//input_angle = sqrt(abs(input_angle)) * glm::sign(input_angle);
	//input_anvel = sqrt(abs(input_anvel)) * glm::sign(input_anvel);
	//input_horzvel = sqrt(abs(input_horzvel)) * glm::sign(input_horzvel);
	//input_vertvel = sqrt(abs(input_vertvel)) * glm::sign(input_vertvel);
	//input_posx = sqrt(abs(input_posx)) * glm::sign(input_posx);
	//input_posy = sqrt(abs(input_posy)) * glm::sign(input_posy);


	net_inputs.clear();
	net_inputs = { input_angle, input_anvel, input_horzvel , input_vertvel, input_posx, input_posy };
	myNet.feedForward(net_inputs);

	//if (target_delta.Length() < 2.0) {
	//	printf("BOOST %f\n", target_delta.Length());
	//}

	//printf("%6.4f, %6.4f\n", input_angle, input_vertvel);

	myNet.getResults(net_outputs);
	boosters(net_outputs[0], net_outputs[1]);

	world.Step(timeStep, velocityIterations, positionIterations);

	glutTimerFunc(1000.0f / framerate, force_redraw, 0);
}

void reshape(int width, int height) {
	SCREEN_W = width;
	SCREEN_H = height;
	ASPECT = (float)SCREEN_W / (float)SCREEN_H;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glLoadIdentity();
	gluOrtho2D(0, SCREEN_W, 0, SCREEN_H);
}

void train() {
	int epochs = 40000;
	int frames = 500;

	float avg_score = 0.0f;
	//float avg_score_count = 0.0f;

	vector<float> scores;

	float max_total_score = 0.0f;
	Net bestNet(topology);


	FILE *log_file;
	fopen_s(&log_file, "trainlog.csv", "w");

	for (int epoch = 0; epoch < epochs; epoch++) {
		// start body center screen with random angle
		body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
		body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

		// rnd ang vel
		double random_val = (((rand() / (double)(RAND_MAX)) * 1.0) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
		//body->SetAngularVelocity(random_val);
		//body->SetAngularVelocity(0.2);

		//float newpos_x = ((rand() / (float)RAND_MAX) * SCREEN_W) / view_scale;
		//float newpos_y = ((rand() / (float)RAND_MAX) * SCREEN_H) / view_scale;

		float rand_angle = (rand() / (double)(RAND_MAX)) * M_PI * 2.0f;
		float rand_radius = ((rand() / (double)(RAND_MAX)) * ((SCREEN_H / 4.0f) / view_scale)) + ((SCREEN_H / 4.0f) / view_scale);

		float newpos_x = glm::cos(rand_angle) * rand_radius + ((SCREEN_W / 2.0f) / view_scale);
		float newpos_y = glm::sin(rand_angle) * rand_radius + ((SCREEN_H / 2.0f) / view_scale);

		body->SetTransform(b2Vec2(newpos_x, newpos_y), 0.0f);
		body->SetAngularVelocity(0.0);

		vector<training_data_entry> current_training;
		//vector<double> scores;
		float score = 0.0f;
		float total_score = 0.0f;

		vector<double> net_inputs;
		vector<double> net_outputs;

		double prev_score = 0.0f; // use for immediate reward
		double max_delta_score = 0.0f;
		double fixed_delta_score = 0.01f;

		double sixth = 1.0 / 6.0;

		bool ignore_epoch = false;
		bool boost = false;

		for (int i = 0; i < frames; i++) {
			// sig 0 -> 0.5 -> 1
			float input_angle = glm::mod((float)body->GetAngle() - M_PI, M_PI * 2.0f) / (M_PI * 2.0f);
			float input_anvel = glm::clamp(body->GetAngularVelocity() + 2.0f, 0.0f, 4.0f) / 4.0f;
			float input_horzvel = glm::clamp(body->GetLinearVelocity().x + 2.0f, 0.0f, 4.0f) / 4.0f;
			float input_vertvel = glm::clamp(body->GetLinearVelocity().y + 2.0f, 0.0f, 4.0f) / 4.0f;

			b2Vec2 pos = body->GetPosition();
			b2Vec2 target(16.0f, 12.0f);
			b2Vec2 target_delta = (target - pos);
			float input_posx = (target_delta.x * view_scale) / (float)SCREEN_W * 2.0;
			float input_posy = (target_delta.y * view_scale) / (float)SCREEN_H * 2.0;
			input_posx = glm::clamp(input_posx + 1.0f, 0.0f, 2.0f) / 2.0f;
			input_posy = glm::clamp(input_posy + 1.0f, 0.0f, 2.0f) / 2.0f;

			// increment score
			//double current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_horzvel - 0.5)) * 2.0) * ((0.5 - abs(input_vertvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			//double current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			//double current_score = ((0.5 - abs(input_angle - 0.5)) + 0.5) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);

			double current_score = max((0.5 - abs(input_angle - 0.5)) * 2.0, 0.9) * max((0.5 - abs(input_anvel - 0.5)) * 2.0, 0.9) * max((0.5 - abs(input_posx - 0.5)) * 2.0, 0.1) * max((0.5 - abs(input_posy - 0.5)) * 2.0, 0.1);
			
			
			//double current_score = ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			
			if (target_delta.Length() < 2.0) {
				boost = true;
			}

			
			// ramp score
			/*
			if (epoch < 1000) {
				current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0);// *((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			}
			else {
				current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) *((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			}
			*/
			/*
			double current_score = 
				sixth * 2.0 * ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0)
				+ sixth * 2.0 * ((0.5 - abs(input_horzvel - 0.5)) * 2.0) * ((0.5 - abs(input_vertvel - 0.5)) * 2.0) 
				+ sixth * 2.0 * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
			*/
			total_score += current_score;

			double delta_score = current_score - prev_score;
			prev_score = current_score;

			score += delta_score;

			// test incentive
			// delta_score += ((delta_score * (1.0 - current_score)) * ((double)i / (double)frames));

			//delta_score += ((delta_score) * ((double)i / (double)frames));
			// todo increase this ^^^ as i increases


			if (i > 0) {
				current_training[i - 1].score_delta = delta_score;

				if (delta_score > max_delta_score) max_delta_score = delta_score;
			}


			// tanh -1 -> 1
			input_angle = (input_angle * 2.0) - 1.0;
			input_anvel = (input_anvel * 2.0) - 1.0;
			input_horzvel = (input_horzvel * 2.0) - 1.0;
			input_vertvel = (input_vertvel * 2.0) - 1.0;
			input_posx = (input_posx * 2.0) - 1.0;
			input_posy = (input_posy * 2.0) - 1.0;

			//squarert everythign
			//input_angle = sqrt(abs(input_angle)) * glm::sign(input_angle);
			//input_anvel = sqrt(abs(input_anvel)) * glm::sign(input_anvel);
			//input_horzvel = sqrt(abs(input_horzvel)) * glm::sign(input_horzvel);
			//input_vertvel = sqrt(abs(input_vertvel)) * glm::sign(input_vertvel);
			//input_posx = sqrt(abs(input_posx)) * glm::sign(input_posx);
			//input_posy = sqrt(abs(input_posy)) * glm::sign(input_posy);

			if (abs(input_angle) > 0.6) {
				//ignore_epoch = true;
				break;
			}


			net_inputs.clear();
			net_inputs = { input_angle, input_anvel, input_horzvel, input_vertvel, input_posx, input_posy };
			myNet.feedForward(net_inputs);

			myNet.getResults(net_outputs);

			if (((rand() % 100) < 4)) {
				double rand_boost_amount = 0.8;

				//if (((rand() % 100) < 20)) {
				//	rand_boost_amount = (glm::clamp(body->GetLinearVelocity().Length(), 0.0f, 4.0f) / 8.0) + 0.5;
				//}

				net_outputs[0] = ((rand() / (double)RAND_MAX) * rand_boost_amount);
				net_outputs[1] = ((rand() / (double)RAND_MAX) * rand_boost_amount);
			}

			boosters(net_outputs[0], net_outputs[1]);

			//training_data_entry current_entry = { input_angle, input_anvel, input_horzvel, net_outputs[0], net_outputs[1], 0.0 };
			training_data_entry current_entry = { input_angle, input_anvel, input_horzvel, input_vertvel, input_posx, input_posy, net_outputs[0], net_outputs[1], 0.0};
			current_training.push_back(current_entry);

			world.Step(timeStep, velocityIterations, positionIterations);
		}

		//avg_score += total_score;
		//avg_score_count += 1.0f;

		scores.push_back(total_score);
		if (scores.size() > 40) {
			scores.erase(scores.begin());
		}
		avg_score = std::accumulate(scores.begin(), scores.end(), 0.0f);
		avg_score /= (float)scores.size();


		vector<double> target_outputs;
		int num_valuable = 0;

		if ((max_delta_score > 0.0) && (!ignore_epoch)) {
			double delta_score_scale = 175.0; // 4000.0 * (total_score / (double)frames); //(50.0 / max_delta_score);
			

			//delta_score_scale *= ((frames - total_score) / (double)frames);

			if (total_score > (avg_score * 0.0)) {
				if (total_score > max_total_score) {
					max_total_score = total_score;
					bestNet = *new Net(myNet);
				}

				for (int t = 1; t < (int)current_training.size(); t++) {
					if (current_training[t].score_delta > (0.7 * max_delta_score)) {
					//if (current_training[t].score_delta > (((boost == true) ? 0.4 : 0.7) * max_delta_score)) {
						num_valuable++;

						int numLoops = (int)glm::min(abs(current_training[t].score_delta * delta_score_scale), 150.0);
						//int numLoops = (int)glm::min(abs(glm::min((double)current_training[t].score_delta, fixed_delta_score) * delta_score_scale), 150.0);

						//int numLoops = ((double)total_score / (double)frames) * 30;
						if (score > 0) {
							//numLoops = (int)((double)numLoops * (1.0 + score));
							//numLoops = (int)((double)numLoops * ((double)total_score / (double)(frames / 10.0)));
						}

						if (boost) {
							//numLoops *= 1.5f;
						}

						for (int i = 0; i < numLoops; i++) {
						//for (int i = 0; i < 50; i++) {
							net_inputs.clear();
							net_inputs = { current_training[t].angle, current_training[t].anvel , current_training[t].horzvel, current_training[t].vertvel, current_training[t].targetdx, current_training[t].targetdy };
							myNet.feedForward(net_inputs);

							target_outputs.clear();
							target_outputs = { current_training[t].boost_left, current_training[t].boost_right };

							if (current_training[t].score_delta > 0) {
								myNet.backProp(target_outputs, true);
							}
						}
					}
				}
			}
		}

		printf("SCORE: %6.3f | MAX D: %6.4f | USED: %3d | TSCORE: %5.1f | ASCORE: %5.1f | EPOCH: %5d", score, max_delta_score, num_valuable, total_score, avg_score, epoch);
		if (boost) printf("*");
		printf("\n");

		fprintf(log_file, "%5d, %5.1f\n", epoch, total_score);
	}

	fclose(log_file);
	//myNet = *new Net(bestNet);

	//printf("AVG SCORE: %f\n", avg_score / avg_score_count);
}

void train_new() {
	int epochs = 4000;
	int frames = 500;
	int num_rewinds = 20;

	float avg_score = 0.0f;

	vector<float> scores;

	float max_total_score = 0.0f;
	Net bestNet(topology);


	FILE *log_file;
	fopen_s(&log_file, "trainlog.csv", "w");

	for (int epoch = 0; epoch < epochs; epoch++) {
		
		vector<vector<training_data_entry>> training_batches;
		vector<float> training_scores;
		vector<float> batches_max_delta_score;
		float highest_training_score = 0.0f;
		float highest_max_delta = 0.0f;
		int training_batch_index = 0;


		float rand_angle = (rand() / (double)(RAND_MAX)) * M_PI * 2.0f;
		float rand_radius = ((rand() / (double)(RAND_MAX)) * ((SCREEN_H / 4.0f) / view_scale)) + ((SCREEN_H / 4.0f) / view_scale);
		//rand_radius /= 2.0f;
		
		//float rand_angle = 0.2345f;
		//float rand_radius = 7.0f;

		float newpos_x = glm::cos(rand_angle) * rand_radius + ((SCREEN_W / 2.0f) / view_scale);
		float newpos_y = glm::sin(rand_angle) * rand_radius + ((SCREEN_H / 2.0f) / view_scale);

		vector<double> net_inputs;
		vector<double> net_outputs;

		for (int rewind = 0; rewind < num_rewinds; rewind++) {
			//body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
			body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

			body->SetTransform(b2Vec2(newpos_x, newpos_y), 0.0f);
			body->SetAngularVelocity(0.0);

			vector<training_data_entry> current_training;
			//vector<double> scores;
			float score = 0.0f;
			float total_score = 0.0f;

			double prev_score = 0.0f; // use for immediate reward
			double max_delta_score = 0.0f;

			double sixth = 1.0 / 6.0;

			bool ignore_epoch = false;

			for (int i = 0; i < frames; i++) {
				// sig 0 -> 0.5 -> 1
				float input_angle = glm::mod((float)body->GetAngle() - M_PI, M_PI * 2.0f) / (M_PI * 2.0f);
				float input_anvel = glm::clamp(body->GetAngularVelocity() + 2.0f, 0.0f, 4.0f) / 4.0f;
				float input_horzvel = glm::clamp(body->GetLinearVelocity().x + 2.0f, 0.0f, 4.0f) / 4.0f;
				float input_vertvel = glm::clamp(body->GetLinearVelocity().y + 2.0f, 0.0f, 4.0f) / 4.0f;

				b2Vec2 pos = body->GetPosition();
				b2Vec2 target(16.0f, 12.0f);
				b2Vec2 target_delta = (target - pos);
				float input_posx = (target_delta.x * view_scale) / (float)SCREEN_W * 2.0;
				float input_posy = (target_delta.y * view_scale) / (float)SCREEN_H * 2.0;
				input_posx = glm::clamp(input_posx + 1.0f, 0.0f, 2.0f) / 2.0f;
				input_posy = glm::clamp(input_posy + 1.0f, 0.0f, 2.0f) / 2.0f;

				// increment score
				//double current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_horzvel - 0.5)) * 2.0) * ((0.5 - abs(input_vertvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
				//double current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);
				//double current_score = ((0.5 - abs(input_angle - 0.5)) + 0.5) * ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);

				double current_score = max((0.5 - abs(input_angle - 0.5)) * 2.0, 0.2) * max((0.5 - abs(input_anvel - 0.5)) * 2.0, 0.2) * max((0.5 - abs(input_posx - 0.5)) * 2.0, 0.1) * max((0.5 - abs(input_posy - 0.5)) * 2.0, 0.1);

				//double current_score = ((0.5 - abs(input_anvel - 0.5)) * 2.0) * ((0.5 - abs(input_posx - 0.5)) * 2.0) * ((0.5 - abs(input_posy - 0.5)) * 2.0);

				total_score += current_score;

				double delta_score = current_score - prev_score;
				prev_score = current_score;

				score += delta_score;

				// test incentive
				// delta_score += ((delta_score * (1.0 - current_score)) * ((double)i / (double)frames));
				delta_score += ((delta_score * (1.0 - current_score)));

				//delta_score += ((delta_score) * ((double)i / (double)frames));
				// todo increase this ^^^ as i increases


				if (i > 0) {
					current_training[i - 1].score_delta = delta_score;

					if (delta_score > max_delta_score) max_delta_score = delta_score;
				}


				// tanh -1 -> 1
				input_angle = (input_angle * 2.0) - 1.0;
				input_anvel = (input_anvel * 2.0) - 1.0;
				input_horzvel = (input_horzvel * 2.0) - 1.0;
				input_vertvel = (input_vertvel * 2.0) - 1.0;
				input_posx = (input_posx * 2.0) - 1.0;
				input_posy = (input_posy * 2.0) - 1.0;

				//squarert everythign
				//input_angle = sqrt(abs(input_angle)) * glm::sign(input_angle);
				//input_anvel = sqrt(abs(input_anvel)) * glm::sign(input_anvel);
				//input_horzvel = sqrt(abs(input_horzvel)) * glm::sign(input_horzvel);
				//input_vertvel = sqrt(abs(input_vertvel)) * glm::sign(input_vertvel);
				//input_posx = sqrt(abs(input_posx)) * glm::sign(input_posx);
				//input_posy = sqrt(abs(input_posy)) * glm::sign(input_posy);

				if (abs(input_angle) > 0.6) {
					//ignore_epoch = true;
					break;
				}


				net_inputs.clear();
				net_inputs = { input_angle, input_anvel, input_horzvel, input_vertvel, input_posx, input_posy };
				myNet.feedForward(net_inputs);

				myNet.getResults(net_outputs);

				if (((rand() % 100) < (2*rewind + 2))) {
					double rand_boost_amount = 0.77;

					net_outputs[0] = ((rand() / (double)RAND_MAX) * rand_boost_amount);
					net_outputs[1] = ((rand() / (double)RAND_MAX) * rand_boost_amount);
				}

				boosters(net_outputs[0], net_outputs[1]);

				//training_data_entry current_entry = { input_angle, input_anvel, input_horzvel, net_outputs[0], net_outputs[1], 0.0 };
				training_data_entry current_entry = { input_angle, input_anvel, input_horzvel, input_vertvel, input_posx, input_posy, net_outputs[0], net_outputs[1], 0.0 };
				current_training.push_back(current_entry);

				world.Step(timeStep, velocityIterations, positionIterations);
			}

			// add stuff

			//vector<vector<training_data_entry>> training_batches;
			//vector<float> training_scores;
			//vector<float> batches_max_delta_score;
			//float highest_training_score = 0.0f;
			//int training_batch_index = 0;

			training_batches.push_back(current_training);
			training_scores.push_back(total_score);
			batches_max_delta_score.push_back(max_delta_score);

			if (total_score > highest_training_score) {
				highest_max_delta = max_delta_score;
				highest_training_score = total_score;
				training_batch_index = rewind;
			}
			//if (max_delta_score > highest_max_delta) {
			//	highest_max_delta = max_delta_score;
			//	highest_training_score = total_score;
			//	training_batch_index = rewind;
			//}
		}




		//avg_score += total_score;
		//avg_score_count += 1.0f;

		scores.push_back(highest_training_score);
		if (scores.size() > 40) {
			scores.erase(scores.begin());
		}
		avg_score = std::accumulate(scores.begin(), scores.end(), 0.0f);
		avg_score /= (float)scores.size();


		vector<double> target_outputs;
		int num_valuable = 0;

		if ((batches_max_delta_score[training_batch_index] > 0.0)) {
			double delta_score_scale = 175.0; // 4000.0 * (total_score / (double)frames); //(50.0 / max_delta_score);


											  //delta_score_scale *= ((frames - total_score) / (double)frames);

			if (highest_training_score > (avg_score * 0.0)) {
				if (highest_training_score > max_total_score) {
					max_total_score = highest_training_score;
					bestNet = *new Net(myNet);
				}

				for (int t = 1; t < (int)training_batches[training_batch_index].size(); t++) {
					if (training_batches[training_batch_index][t].score_delta >(0.7 * batches_max_delta_score[training_batch_index])) {
						//if (current_training[t].score_delta > (((boost == true) ? 0.4 : 0.7) * max_delta_score)) {
						num_valuable++;

						int numLoops = (int)glm::min(abs(training_batches[training_batch_index][t].score_delta * delta_score_scale), 150.0);
						//int numLoops = (int)glm::min(abs(glm::min((double)current_training[t].score_delta, fixed_delta_score) * delta_score_scale), 150.0);

						//int numLoops = ((double)total_score / (double)frames) * 30;


						for (int i = 0; i < numLoops; i++) {
							//for (int i = 0; i < 50; i++) {
							net_inputs.clear();
							net_inputs = { training_batches[training_batch_index][t].angle, training_batches[training_batch_index][t].anvel , training_batches[training_batch_index][t].horzvel, training_batches[training_batch_index][t].vertvel, training_batches[training_batch_index][t].targetdx, training_batches[training_batch_index][t].targetdy };
							myNet.feedForward(net_inputs);

							target_outputs.clear();
							target_outputs = { training_batches[training_batch_index][t].boost_left, training_batches[training_batch_index][t].boost_right };

							if (training_batches[training_batch_index][t].score_delta > 0) {
								myNet.backProp(target_outputs, true);
							}
						}
					}
				}
			}
		}

		printf("SCORE: %6.3f | MAX D: %6.4f | USED: %3d | TSCORE: %5.1f | ASCORE: %5.1f | EPOCH: %5d\n", 0.0, batches_max_delta_score[training_batch_index], num_valuable, highest_training_score, avg_score, epoch);

		fprintf(log_file, "%5d, %5.1f\n", epoch, highest_training_score);
	}

	fclose(log_file);
	//myNet = *new Net(bestNet);

	//printf("AVG SCORE: %f\n", avg_score / avg_score_count);
}

int main(int argc, const char * argv[]) {
	srand(1337);


	glutInit(&argc, const_cast<char**>(argv));

	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA);
	glutInitWindowSize(SCREEN_W, SCREEN_H);
	glutCreateWindow("Dumb Rockets");
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	GLuint error = glewInit();

	glutDisplayFunc(render);
	glutTimerFunc(20, force_redraw, 0);
	glutKeyboardFunc(keyboard_down);
	glutKeyboardUpFunc(keyboard_up);
	glutSpecialFunc(keyboard_down_special);
	glutSpecialUpFunc(keyboard_up_special);
	glutReshapeFunc(reshape);

	glMatrixMode(GL_PROJECTION);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	init_world();

	train_new();

	body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
	body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

	// rnd ang vel
	double random_val = (((rand() / (double)(RAND_MAX)) * 0.5) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
	body->SetAngularVelocity(random_val);

	glutMainLoop();

	return 0;
}
