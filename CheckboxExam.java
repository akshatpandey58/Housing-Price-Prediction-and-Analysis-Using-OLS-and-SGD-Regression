import java.awt.*;    
import java.awt.event.*;    
public class CheckboxExample2  
{      
     CheckboxExample2() {      
        Frame f = new Frame ("CheckBox Example");     
        final Label label1 = new Label(); 
	final Label label2 = new Label(); 
	label1.setBounds(120,90,200,10);
	label2.setBounds(120,170,200,10);
        Checkbox checkbox1 = new Checkbox("C++");    
        checkbox1.setBounds(150, 100,50, 50);    
        Checkbox checkbox2 = new Checkbox("Java");    
        checkbox2.setBounds(150, 180,50, 50);  
	f.add(checkbox1);  
	f.add(checkbox2);   
	f.add(label1);
	f.add(label2);
	 
      checkbox1.addItemListener(new ItemListener() {    
             public void itemStateChanged(ItemEvent e) {                 
                label1.setText("C++ Checkbox: "     
                + (e.getStateChange()==1?"checked":"unchecked"));    
             }    
          });    
        checkbox2.addItemListener(new ItemListener() {    
             public void itemStateChanged(ItemEvent e) {                 
                label2.setText("Java Checkbox: "     
                + (e.getStateChange()==1?"checked":"unchecked"));    
             }    
          });     
        f.setSize(400,400);    
        f.setLayout(null);    
        f.setVisible(true);    
     }    
public static void main(String args[])    
{    
    new CheckboxExample2();    
}    
}  
