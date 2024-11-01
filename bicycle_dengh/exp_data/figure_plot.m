% Read the CSV file with the original column names preserved
opts = detectImportOptions('roll_angle_data.csv', 'VariableNamingRule', 'preserve');
data = readtable('roll_angle_data.csv', opts);

% Extract step column
steps = data.Step;

% Extract multiple roll angle columns with preserved names
roll_angle_0 = data.('Roll Angle 0 degree_1');
roll_angle_3 = data.('Roll Angle 3 degree_1');
roll_angle_5 = data.('Roll Angle 5 degree_1');
roll_angle_7 = data.('Roll Angle 7 degree_1');

% Plot the data
figure;
plot(steps, roll_angle_0, 'k-', 'LineWidth', 1.5); hold on;
plot(steps, roll_angle_3, 'r-', 'LineWidth', 1.5);
plot(steps, roll_angle_5, 'g-', 'LineWidth', 1.5);
plot(steps, roll_angle_7, 'b-', 'LineWidth', 1.5);

% Add labels, title, and legend
xlabel('Step');
ylabel('Roll Angle (degree)');
% title('Comparison of balance for different values of initial roll angle');
legend({'0 degrees', '3 degrees', '5 degrees', '7 degrees'}, 'Location', 'best');
grid on;
hold off;

