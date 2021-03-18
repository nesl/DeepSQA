train_x = [data_s_test, data_q_test ]
# train_x = [data_q_valid]
train_y = data_a_test
    
trained_model_1 = load_model('616_baseline_models_s4/cnn_lstm_cat.hdf5')
# print('Model loaded: ', model_name)

# evaluate saved model
y_pred1 = trained_model_1.predict(train_x, batch_size=64, verbose=1)
y_pred1= np.argmax(y_pred1, axis=1)
y_true1 = np.argmax(train_y, axis=1)

y_ans = (y_pred1==y_true1)
print(sum(y_ans)/y_ans.shape)



import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    
# class_names = np.array(range(25))
class_names = tt_ans

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true1, y_pred1)
np.set_printoptions(precision=2)

print(sum(y_true1==y_pred1)/y_true1.shape[0])

# Plot non-normalized confusion matrix
plt.figure(figsize=(20,20))
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



tt_ans = ['no',
 'yes',
 '0',
 '1',
 '2',
 'open the first drawer',
 'open the back door',
 'close the first drawer',
 'close the third drawer',
 'close the dishwasher',
 'open the fridge',
 'open the dishwasher',
 'drink from the cup',
 'close the front door',
 'close the fridge',
 'close the back door',
 'toggle the switch',
 'open the front door',
 'open the third drawer',
 'close the second drawer',
 'open the second drawer',
 '3',
 'clean the table',
 '4']