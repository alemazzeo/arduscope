#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif

//#define DEBUG

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

// # MACRO info_print()
#define info_print(...) \
    do { if (DEBUG_TEST) {\
            Serial.print("INFO : (");\
            Serial.print(__LINE__);\
            Serial.print(") : ");\
            Serial.println(__VA_ARGS__);}\
      } while (0)

// # MACRO info_print_var()
#define info_print_var(NAMEVAR, VALUE) \
    do { if (DEBUG_TEST) {\
            Serial.print("INFO : (");\
            Serial.print(__LINE__);\
            Serial.print(") : ");\
            Serial.print(NAMEVAR);\
            Serial.print(" = ");\
            Serial.println(VALUE);}\
      } while (0)


#ifdef DEBUG
#define BUFFER_SIZE 10
#else
#define BUFFER_SIZE 480
#endif

#define PORTD_OUTPUT_PIN 7

int buffer_write = 0;
int buffer_read = 0;
int trigger_position = 0;
bool data_ready = false;
int transmited = 0;

int limit = 0;
int screens = 0;

int pulse_width = 1000;
int timer_2_counter = 0;

int trigger_channel = 0;
int n_channels = 3;

int last_trigger_data = 0;
int trigger = 0;
int trigger_offset = 0;
int trigger_tol = 5;

bool trigger_min = false;
bool triggered = false;

int new_data = 0;

unsigned int buffer[BUFFER_SIZE];
int params[10];


void set_timer_1(int frequency){

    int prescaler_level = 0;
    int prescaler_values[] = {1, 8, 64, 256, 1025};

    long cmr = 16000000 / (prescaler_values[prescaler_level] * frequency) - 1;

    while (cmr >= 65536) {
        if (prescaler_level == 5){
            Serial.println("Invalid frequency");
            return;
        }
        prescaler_level += 1;
        cmr = 16000000 / (prescaler_values[prescaler_level] * frequency) - 1;
    }

    TCCR1A = 0;
    TCCR1B = 0;
    TCNT1  = 0;
    OCR1A = cmr;
    TCCR1B |= (1 << WGM12);
    TCCR1B = (TCCR1B & 0b11111000) + (prescaler_level + 1);
}

void set_timer_2(){

    TCCR2A = 0;
    TCCR2B = 0;
    TCNT2  = 0;
    OCR2A = 249;
    TCCR2A |= (1 << WGM21);
    TCCR2B |= (1 << CS22);
}

void apply_setup(
    int _limit,
    int frequency,
    int reference,
    int _trigger,
    int _trigger_channel,
    int _trigger_offset,
    int _trigger_tol,
    int _n_channels,
    int adc_prescaler,
    int _pulse_width
){

    ADCSRA = (ADCSRA & B11111000) | adc_prescaler;

    switch (reference) {
        case 0:
            analogReference(DEFAULT);
            break;
        case 1:
            analogReference(INTERNAL);
            break;
        case 2:
            analogReference(EXTERNAL);
            break;
    }

    n_channels = _n_channels;
    trigger_channel = _trigger_channel;

    trigger = _trigger;
    trigger_offset = _trigger_offset;
    trigger_tol = _trigger_tol;
    limit = _limit;
    screens = 0;
    pulse_width = _pulse_width;

    if (trigger_channel > 0){
        triggered = false;
    } else if (trigger_channel == -3) {
        triggered = true;
        trigger_position = buffer_write;
        buffer_read = 0;
    }

    set_timer_1(frequency);
    set_timer_2();

}


void init_capture(){
    buffer_write = 0;
    buffer_read = 0;
    transmited = 0;
    TIMSK1 |= (1 << OCIE1A);
    TIMSK2 |= (1 << OCIE2A);
}

void stop_capture(){
    TIMSK1 &= ~(1 << OCIE1A);
    TIMSK2 &= ~(1 << OCIE2A);
}

ISR(TIMER1_COMPA_vect) {
    if (data_ready == false) {
        info_print("ACQUIRE");

        for (int i = 0; i<n_channels; i++){
            info_print_var("buffer_write", buffer_write);
            info_print_var("buffer_read", buffer_read);
            info_print_var("channel", i);
            buffer[buffer_write] = analogRead(A0 + i);
            info_print_var("buffer[buffer_write]", buffer[buffer_write]);
            last_trigger_data = buffer[buffer_write];
            if (trigger_channel == i){
                if (!triggered){
                    if (!trigger_min && buffer[buffer_write] < trigger - trigger_tol){
                        trigger_min = true;
                    }

                    if (!triggered &&
                        trigger_min && buffer[buffer_write] > trigger + trigger_tol &&
                        new_data > trigger_offset
                       ) {
                        info_print_var("TRIGGERED! CHANNEL", i);
                        triggered = true;
                        trigger_position = (BUFFER_SIZE + buffer_write - i - trigger_offset * n_channels) % BUFFER_SIZE;
                        buffer_read = trigger_offset;
                    }
                }
            }
            buffer_write = (buffer_write + 1) % BUFFER_SIZE;
        }

        if (!triggered &&
            (
              (trigger_channel == -1 && PORTD & (1 << PORTD_OUTPUT_PIN)) ||
              (trigger_channel == -2 && !(PORTD & (1 << PORTD_OUTPUT_PIN)))
            ) &&
            new_data > trigger_offset
           ) {
            info_print("TRIGGER BY PIN7");
            triggered = true;
            trigger_position = (BUFFER_SIZE + buffer_write - trigger_offset * n_channels) % BUFFER_SIZE;
            info_print_var("trigger_offset", trigger_offset);
            buffer_read = trigger_offset;
        }

        if (triggered) {
            buffer_read += 1;
        }

        if (buffer_read >= BUFFER_SIZE / n_channels){
            info_print_var("DATA READY. READ POSITION", trigger_position);
            data_ready = true;
            triggered = false;
            trigger_min = false;
            buffer_read = 0;
            new_data = 0;
        }


        new_data += 1;

    }
}

ISR(TIMER2_COMPA_vect) {
    timer_2_counter += 1;
    if (timer_2_counter >= pulse_width){
        timer_2_counter = 0;
        PORTD ^= (1 << PORTD_OUTPUT_PIN);
    }
}


void setup() {

    Serial.begin(115200);
    pinMode(13, OUTPUT);
    pinMode(7, OUTPUT);
    pinMode(A0, INPUT);
    pinMode(A1, INPUT);
    pinMode(A2, INPUT);
    pinMode(A3, INPUT);
    pinMode(A4, INPUT);
    pinMode(A5, INPUT);
    sbi(ADCSRA, ADEN); // habilita la conversión ADC
    Serial.println("BOOTED");


#ifdef DEBUG
    apply_setup(
        0, // limit
        1, // frequency
        0, // reference
        512, // trigger
        -1, // trigger_channel
        2, // trigger_offset
        5, // trigger_tol
        2, // channels
        4, // adc_prescaler
        500  // pulse_width
    );

    init_capture();
#endif

}

void loop() {

#ifndef DEBUG
    if (Serial.available() == 20){
        Serial.readBytes((char *) params, sizeof(params));

        apply_setup(
            params[0], // limit
            params[1], // frequency
            params[2], // reference
            params[3], // trigger
            params[4], // trigger_channel
            params[5], // trigger_offset
            params[6], // trigger_tol
            params[7], // channels
            params[8], // adc_prescaler
            params[9]  // pulse_width
        );

        init_capture();
    }
#endif

    if (data_ready) {
        if (limit == 0 || screens < limit){
            screens += 1;
            PORTB ^= (1 << 5);

            Serial.write(trigger_position & 0xFF);
            Serial.write(trigger_position >> 8);
            Serial.write((uint8_t*)buffer, sizeof(buffer));

            data_ready = false;

            if (trigger_channel == -3) {
                triggered = true;
                buffer_write = 0;
                trigger_position = buffer_write;
                buffer_read = 0;
            }
        } else {
            data_ready = false;
            stop_capture();
        }
    }
}
