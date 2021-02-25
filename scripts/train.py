
import tensorflow as tf

def trainModel(tagpath,model,train_x,train_y,batch_size,maxperiod,val_x,val_y):
    log_dir = os.path.join(tagpath,'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    checkpoint_save_path = os.path.join(tagpath,"checkpoint","save.ckpt"
    
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
            
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                    save_weights_only=False,
                                                    save_best_only=False)
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)
        
    history = model.fit(
                        train_x, train_y, 
                        batch_size=batch_size, 
                        epochs=maxperiod, 
                        #steps_per_epoch=10,
                        #validation_steps=10,
                        validation_data=(val_x, val_y), 
                        validation_freq=1,
                        callbacks=[cp_callback,tensorboard])