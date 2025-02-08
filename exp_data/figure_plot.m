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
figure('Position', [100, 100, 800, 370]); 
plot(steps, roll_angle_0, 'k-', 'LineWidth', 1.5); hold on;
plot(steps, roll_angle_3, 'r-', 'LineWidth', 1.5);
plot(steps, roll_angle_5, 'g-', 'LineWidth', 1.5);
plot(steps, roll_angle_7, 'b-', 'LineWidth', 1.5);

% Add labels, title, and legend
xlabel('步数', 'FontSize', 18);
ylabel('倾斜角/度', 'FontSize', 18);
% title('Comparison of balance for different values of initial roll angle');
legend({'0度', '3度', '5度', '7度'}, 'Location', 'best', 'FontSize', 12);
grid on;
hold off;
print('myfigure','-dsvg');  % 导出为 SVG
