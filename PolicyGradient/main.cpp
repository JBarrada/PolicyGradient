#include <glew.h>
#include <glut.h>

#include <vector>
#include <cmath>

#include "glm\glm.hpp"
#include "glm\gtx\rotate_vector.hpp"

#include "Box2D\Box2D.h"

#include "NN.h"

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

b2Vec2 gravity(0.0f, 0.0f);
b2World world(gravity);

b2Body* body;


float boosters_vals[2] = { 0, 0 };

vector<unsigned> topology = { 2, 6, 2 };
Net myNet(topology);

int current_time = 0;
int max_time = 1000;

struct training_data_entry {
	float angle, anvel;
	float boost_left, boost_right;

	float score_delta;
};

vector<training_data_entry> training_data;

vector<training_data_entry> self_training_data;

void init_world() {
	b2BodyDef groundBodyDef;
	groundBodyDef.position.Set(0.0f, -10.0f);

	// Call the body factory which allocates memory for the ground body
	// from a pool and creates the ground box shape (also from a pool).
	// The body is also added to the world.
	b2Body* groundBody = world.CreateBody(&groundBodyDef);

	// Define the ground box shape.
	b2PolygonShape groundBox;

	// The extents are the half-widths of the box.
	groundBox.SetAsBox(50.0f, 10.0f);

	// Add the ground fixture to the ground body.
	groundBody->CreateFixture(&groundBox, 0.0f);

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
	
}

void keyboard_up(unsigned char key, int x, int y) {
	if (key == ' ') {
		body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
		body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

		// rnd ang vel
		double random_val = (((rand() / (double)(RAND_MAX)) * 0.5) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
		body->SetAngularVelocity(random_val);
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

	float input_angle = glm::mod((float)body->GetAngle() - M_PI, M_PI * 2.0f) / (M_PI * 2.0f);
	float input_anvel = glm::clamp(body->GetAngularVelocity() + 2.0f, 0.0f, 4.0f) / 4.0f;

	vector<double> net_inputs;
	vector<double> net_outputs;

	net_inputs.clear();
	net_inputs = { input_angle, input_anvel };
	myNet.feedForward(net_inputs);

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
	int epochs = 500;
	int frames = 100;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// start body center screen with random angle
		body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
		body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

		// rnd ang vel
		double random_val = (((rand() / (double)(RAND_MAX)) * 0.5) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
		//body->SetAngularVelocity(random_val);
		body->SetAngularVelocity(1.2);

		vector<training_data_entry> current_training;
		vector<double> scores;
		float score = 0.0f;

		vector<double> net_inputs;
		vector<double> net_outputs;

		double prev_score = 0.0f; // use for immediate reward
		double max_delta_score = 0.0f;

		for (int i = 0; i < frames; i++) {
			float input_angle = glm::mod((float)body->GetAngle() - M_PI, M_PI * 2.0f) / (M_PI * 2.0f);
			float input_anvel = glm::clamp(body->GetAngularVelocity() + 2.0f, 0.0f, 4.0f) / 4.0f;

			// increment score
			double current_score = ((0.5 - abs(input_angle - 0.5)) * 2.0) * ((0.5 - abs(input_anvel - 0.5)) * 2.0);
			//score += current_score;

			double delta_score = current_score - prev_score;
			prev_score = current_score;

			score += delta_score;

			if (i > 0) {
				current_training[i - 1].score_delta = delta_score;

				if (abs(delta_score) > max_delta_score) max_delta_score = abs(delta_score);
			}

			net_inputs.clear();
			net_inputs = { input_angle, input_anvel };
			myNet.feedForward(net_inputs);

			myNet.getResults(net_outputs);
			boosters(net_outputs[0], net_outputs[1]);

			training_data_entry current_entry = { input_angle, input_anvel, net_outputs[0], net_outputs[1], 0.0};
			current_training.push_back(current_entry);

			world.Step(timeStep, velocityIterations, positionIterations);
		}

		vector<double> target_outputs;
		/*
		for (int i = 0; i < (int)(score / 5.0); i++) {
			for (int t = 0; t < (int)current_training.size(); t++) {
				net_inputs.clear();
				net_inputs = { current_training[t].angle, current_training[t].anvel };
				myNet.feedForward(net_inputs);

				target_outputs.clear();
				target_outputs = { current_training[t].boost_left, current_training[t].boost_right };
				myNet.backProp(target_outputs);
			}
		}
		*/
		int num_valuable = 0;

		if (max_delta_score > 0) {
			double delta_score_scale = 2000.0; //(50.0 / max_delta_score);


			for (int t = 0; t < (int)current_training.size(); t++) {
				//if (abs(current_training[t].score_delta) > (0.9 * max_delta_score)) {
					num_valuable++;
					for (int i = 0; i < (int)glm::min(abs(current_training[t].score_delta * delta_score_scale), 400.0); i++) {
					//for (int i = 0; i < 100; i++) {
						net_inputs.clear();
						net_inputs = { current_training[t].angle, current_training[t].anvel };
						myNet.feedForward(net_inputs);

						target_outputs.clear();
						target_outputs = { current_training[t].boost_left, current_training[t].boost_right };
						/*
						if (current_training[t].score_delta > 0) {
							myNet.backProp(target_outputs, 1.0);
						}
						else {
							myNet.backProp(target_outputs, -1.0);
						}
						*/
						//myNet.backProp(target_outputs, 1.0);
						myNet.backProp(target_outputs, current_training[t].score_delta);
					}
				//}
			}
		}

		printf("SCORE: %f | MAX DELTA: %f | USED: %d\n", score, max_delta_score, num_valuable);
	}

}

int main(int argc, const char * argv[]) {
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

	train();

	body->SetTransform(b2Vec2(16.0, 12.0), 0.0f);
	body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

	// rnd ang vel
	double random_val = (((rand() / (double)(RAND_MAX)) * 0.5) + 0.5) * (((rand() / (double)(RAND_MAX)) > 0.5) ? -1.0 : 1.0);
	body->SetAngularVelocity(random_val);

	glutMainLoop();

	return 0;
}
