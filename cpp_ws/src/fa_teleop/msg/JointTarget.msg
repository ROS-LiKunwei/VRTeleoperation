# Versioned interface: incompatible bundles never exchange motion commands.
std_msgs/Header header
uint64 sequence
string release_id
# 0=controlled stop, 1=initialization, 2=tracking
uint8 mode
float64[7] left
float64[7] right
float64[2] neck
float64 duration_s
float64 velocity_limit
float64 acceleration_limit
